"""This implements a SMA crossover trading strategy in the context of MicroFE
Moving averages are implemented as EMA inline with STC (Shaff Trend Cycle) for convenience. see https://www.tradingpedia.com/forex-trading-indicators/schaff-trend-cycle for more info on STC.

The basic premise is that a trading signal occurs when a short-term moving average (SMA) crosses through a long-term moving average (LMA). Buy signals occur when the SMA crosses above the LMA and a sell signal occurs during the opposite movement.

In finance, trade signals provide entry and exit points, likely at current market price. This is fine for BESS as the purpose is similar (arbitrage). 

For PV trading some sort of dynamic pricing method needs to be used.
BESS charge/discharge tracks PV price EMAs. For simplicity reasons, the dispatchable pool will not be used.
Instead, all bids and asks will go into the PV pool and BESS compensation will be used for discharging

15 step time windows
initialize with randomized prices for each window

trade with price in time step, then calculate per unit profit/cost for each time step
- if bid price too high, then the profit is equal to net metering (low),
- next time try lower price

calculate per unit cost for each time step
- if ask price too low, cost equal net metering (high)
- next time try higher price

"""
import tenacity
from _agent._utils.metrics import Metrics
import asyncio
from _utils import utils
from _utils import jkson as json
from _agent._components import rewards
import random

import sqlalchemy
from sqlalchemy import MetaData, Column
from _utils import db_utils
import databases
import ast
import collections

class EMA:
    def __init__(self, window_size):
        self.window_size = window_size
        self.count = 0
        self.last_average = 0

    def update(self, new_value):
        # approximate EMA
        self.count = min(self.count + 1, self.window_size)
        average = self.last_average + (new_value - self.last_average) / self.count
        self.last_average = average

    def reset(self):
        self.count = 0
        self.last_average = 0

class Trader:
    """This trader uses SMA crossover to make trading decisions in the context of MicroFE

    The trader tries to learn the right prices for each minute of the day. This is done by initializing two prices tables, one for bid prices and one for ask prices. Each table is 1440 elements long. The tables are initialized by randomizing prices of each minute within a price range. A 15 minute window is used for initialization, which means that only 96 initial prices are generated. This is meant to decrease initial noise. Successful trades will nudge bid and ask prices to the point of most profit and least cost.
    """
    def __init__(self, bid_price, ask_price, **kwargs):
        # Some utility parameters
        self.__participant = kwargs['trader_fns']
        self.status = {
            'weights_loading': False,
            'weights_loaded': False,
            'weights_saving': False,
            'weights_saved': True
        }

        self.__db = {
            'path': '',
            'db': None
        }

        # Initialize the agent learning parameters
        self.agent_data = {}
        scale_factor = 1
        self.sma_bid = EMA(23 * scale_factor) #23-period EMA for bid prices
        self.lma_bid = EMA(50 * scale_factor) #50-period EMA for bid prices

        self.sma_ask = EMA(23 * scale_factor)  # 23-period EMA for ask prices
        self.lma_ask = EMA(50 * scale_factor)  # 50-period EMA for ask prices

        # buy signal triggers when sma < lma (bid prices are rapidly dropping, therefore supply increasing)
        # sell signal triggers when sma > lma (ask prices are rapidly increasing, therefore demand increasing)
        self.buy_trigger = False
        self.sell_trigger = False

        self.price_adj_step = 0.001 # next price is adjusted in half-cent steps

        self.bid_price = bid_price
        self.ask_price = ask_price

        # Initialize price tables
        self.bid_prices = self.__generate_price_table(bid_price, ask_price, 15)
        self.ask_prices = self.__generate_price_table(bid_price, ask_price, 15)

        # Initialize learning parameters
        self.learning = kwargs['learning'] if 'learning' in kwargs else False
        self._rewards = rewards.UnitProfitAndCost(self.__participant['timing'],
                                                  self.__participant['ledger'],
                                                  self.__participant['market_info'])

        # Initialize metrics tracking
        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()

    def __init_metrics(self):
        import sqlalchemy
        '''
        Initializes metrics to record into database
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)
        if 'storage' in self.__participant:
            self.metrics.add('storage_soc', sqlalchemy.Float)

    def __generate_price_table(self, bid_price, ask_price, window_size):
        window_price = [random.randint(*sorted([bid_price*100, ask_price*100])) for i in range(int(1440/window_size))]
        price_table = dict(zip(range(1440), [[window_price[int(i/window_size)]/100, 0] for i in range(1440)]))
        return price_table

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):
        if not self.learning:
            return

        rewards = await self._rewards.calculate()
        if rewards == None:
            return

        last_deliver = self.__participant['timing']['last_deliver']
        local_time = utils.timestamp_to_local(last_deliver[1], self.__participant['timing']['timezone'])
        time_index = int(local_time.hour * 60 + local_time.minute)

        # update price table
        # [unit_profit, unit_profit_diff, unit_cost, unit_cost_diff]
        unit_profit = rewards[0]
        unit_profit_diff = rewards[1]
        unit_cost = rewards[2]
        unit_cost_diff = rewards[3]

        last_ask_price = self.ask_prices[time_index][0]
        last_bid_price = self.bid_prices[time_index][0]

        if unit_profit_diff <= 0:
            self.ask_prices[time_index][0] = max(unit_profit, last_ask_price-self.price_adj_step)
        else:
            if unit_profit_diff - self.ask_prices[time_index][1] > 0:
                self.ask_prices[time_index][0] += self.price_adj_step
            else:
                self.ask_prices[time_index][0] -= self.price_adj_step
        self.ask_prices[time_index][1] = unit_profit_diff

        if unit_cost_diff >= 0:
            self.bid_prices[time_index][0] = min(unit_cost, last_bid_price + self.price_adj_step)
        else:
            if unit_cost_diff - self.bid_prices[time_index][1] > 0:
                self.bid_prices[time_index][0] -= self.price_adj_step
            else:
                self.bid_prices[time_index][0] += self.price_adj_step
        self.bid_prices[time_index][1] = unit_cost_diff

        # update EMA trackers
        self.sma_bid.update(self.bid_prices[time_index][0])
        self.lma_bid.update(self.bid_prices[time_index][0])

        self.sma_ask.update(self.ask_prices[time_index][0])
        self.lma_ask.update(self.ask_prices[time_index][0])

    async def act(self, **kwargs):
        # ACTIONS ARE FOR THE NEXT SETTLE!!!!!
        # actions are defined by a dictionary in the following format:
        # see participant code for more details.

        # actions = {
        #     'bess': {
        #         time_interval: scheduled_qty. +for charge, -for discharge
        #     },
        #     'bids': {
        #         time_interval: {
        #             'quantity': qty,
        #             'source': source,
        #             'price': dollar_per_kWh
        #         }
        #     },
        #     'asks': {
        #         time_interval: {
        #             'quantity': qty,
        #             'source': source,
        #             'price': dollar_per_kWh
        #         }
        #     }
        # }

        actions = {}
        next_settle = self.__participant['timing']['next_settle']
        grid_prices = self.__participant['market_info'][str(next_settle)]['grid']

        local_time = utils.timestamp_to_local(next_settle[1], self.__participant['timing']['timezone'])
        time_index = int(local_time.hour * 60 + local_time.minute)
        next_generation, next_load = await self.__participant['read_profile'](next_settle)
        next_residual_load = next_load - next_generation
        next_residual_gen = -next_residual_load

        if 'storage' in self.__participant:
            # Adjust how much to charge or discharge based on last successful trade
            current_round = self.__participant['timing']['current_round']
            ls_generation, ls_load = await self.__participant['read_profile'](current_round)
            ls_residual_load = ls_load - ls_generation
            ls_storage_schedule = await self.__participant['storage']['check_schedule'](current_round)
            ls_storage_scheduled = ls_storage_schedule[current_round]['energy_scheduled']
            settled = self.__participant['ledger'].get_settled_info(current_round)

            if ls_storage_scheduled > 0:
                new_quantity = min(ls_storage_scheduled,
                                   max(0, settled['bids']['quantity'] - max(0, ls_residual_load)))
                actions['bess'] = {str(current_round): new_quantity}
            elif ls_storage_scheduled < 0:
                new_quantity = ls_storage_scheduled - settled['asks']['quantity']
                actions['bess'] = {str(current_round): new_quantity}

            storage_schedule = await self.__participant['storage']['check_schedule'](next_settle)
            projected_soc = storage_schedule[next_settle]['projected_soc_end']

        if next_residual_load > 0:
            if 'storage' in self.__participant:
                max_charge = storage_schedule[next_settle]['energy_potential'][1]
                max_discharge = storage_schedule[next_settle]['energy_potential'][0]
                effective_discharge = max(-max(0, next_residual_load), max_discharge)
                residual_discharge = max_discharge - effective_discharge

                if (self.sma_bid.last_average < self.lma_bid.last_average) and projected_soc < 0.9:
                    self.buy_trigger = True
                else:
                    self.buy_trigger = False

                if (self.sma_ask.last_average >= self.lma_ask.last_average) and projected_soc > 0.3:
                    self.sell_trigger=True
                else:
                    self.sell_trigger=False

                if self.buy_trigger and self.sell_trigger:
                    self.buy_trigger = False
                    self.sell_trigger = True


                if 'bess' in actions:
                    actions['bess'][str(next_settle)] = effective_discharge
                else:
                    actions['bess'] = {str(next_settle): effective_discharge}
            else:
                max_charge = 0
                effective_discharge = 0
                residual_discharge = 0

            if self.buy_trigger:
                actions['bids'] = {
                    str(next_settle): {
                        'quantity': next_residual_load + max_charge,
                        'source': 'solar',
                        'price': self.bid_prices[time_index][0]
                    }
                }
                if 'bess' in actions:
                    actions['bess'][str(next_settle)] = max_charge
                else:
                    actions['bess'] = {str(next_settle): max_charge}
            else:
                final_residual_load = next_residual_load + effective_discharge
                if final_residual_load > 0:
                    actions['bids'] = {
                        str(next_settle): {
                            'quantity': final_residual_load,
                            'source': 'solar',
                            'price': self.bid_prices[time_index][0]
                        }
                    }
                
                # comment out this else statement to disable selling stored enegy back to the market
                else:
                    # there is excess discharge capacity to spare
                    if residual_discharge < 0 and self.sell_trigger:
                        actions['asks'] = {
                            str(next_settle): {
                                'quantity': abs(residual_discharge),
                                'source': 'solar',
                                'price': self.ask_prices[time_index][0]
                            }
                        }

        elif next_residual_gen > 0:
            # this version of the SMA agent does not account for the case when both solar and storage are available
            # if storage is available, it is ignored here.
            final_residual_gen = next_residual_gen
            if final_residual_gen > 0:
                actions['asks'] = {
                    str(next_settle): {
                        'quantity': final_residual_gen,
                        'source': 'solar',
                        'price': self.ask_prices[time_index][0]
                    }
                }

        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', next_load),
                self.metrics.track('next_settle_generation', next_generation))
            if 'storage' in self.__participant:
                await self.metrics.track('storage_soc', projected_soc)

            await self.metrics.save(10000)
        return actions

    async def save_weights(self, **kwargs):
        '''
        Save the price tables at the end of the episode into database
        '''

        if 'validation' in kwargs['market_id']:
            return True

        self.status['weights_saving'] = True
        self.status['weights_saved'] = False

        weights = [{
            'generation': kwargs['generation'],
            'bid_prices': str(self.bid_prices),
            'ask_prices': str(self.ask_prices)
        }]

        asyncio.create_task(db_utils.dump_data(weights, kwargs['output_db'], self.__db['table']))
        self.status['weights_saving'] = False
        self.status['weights_saved'] = True
        return True


    async def reset(self, **kwargs):
        self.lma_bid.reset()
        self.lma_ask.reset()
        self.sma_bid.reset()
        self.sma_ask.reset()
        return True