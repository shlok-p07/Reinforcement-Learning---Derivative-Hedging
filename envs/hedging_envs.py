'''Delta Hedging Environment'''
from typing import Dict
import numpy as np

class DeltaHedgingEnv:
    '''
    Simulates a delta-hedged portfolio for a short European call option.
    '''
    def __init__(self, simulator, option,maturity:float,dt:float,trans_cost:float=0.0) -> None:
        self.simulator = simulator
        self.option = option
        self.maturity = maturity
        self.dt = dt
        self.trans_cost = trans_cost
        self.portfolio_val = 0.0
        self.reset()
    def reset(self) -> Dict[str,float]:
        '''resets the environment to initial state'''
        self.tau = self.maturity
        self.spot = self.simulator.reset()
        self.option_price = self.option.price(self.spot,self.tau)
        self.delta = self.option.delta(self.spot, self.tau)
        self.stock_position = self.delta
        self.cash = self.option_price - self.stock_position * self.spot
        return self.get_state()
    def step(self) -> Dict[str,float]:
        '''updates the environment by one tick'''
        self.spot = self.simulator.step()
        self.tau -= self.dt
        self.option_price = self.option.price(self.spot,self.tau)
        delta_new = self.option.delta(self.spot,self.tau)
        trade = delta_new - self.stock_position
        trading_val = abs(trade) * self.spot
        self.cash -= trade * self.spot
        self.cash -= trading_val
        self.stock_position = delta_new
        self.delta = delta_new
        self.portfolio_val = self.computeportfolio_val()
        return self.get_state()

    def computeportfolio_val(self) -> float:
        '''
        computing the portfolio value by adding stock + cash - option liability
        '''
        return self.stock_position * self.spot + self.cash - self.option_price
    def get_state(self) -> Dict[str,float]:
        '''
        Current observable state.
        '''
        return {
            "spot":self.spot,
            "tau":self.tau,
            "option_price":self.option_price,
            "delta":self.delta,
            "stock_position":self.stock_position,
            "cash": self.cash,
            "portfolio_val":self.portfolio_val,
        }
