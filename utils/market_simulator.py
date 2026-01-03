'''
Market Simulator file
'''
from typing import Optional
import numpy as np
class Simulator:
    '''Geometric Brownian Motion simulator for asset prices.'''
    def __init__(self, s0: float, mu: float, sigma: float, dt: float, seed: Optional[int]):
        '''
        Initializing simulator variables per the Geometric Brownian Motion model.
        '''
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.reset()
    def reset(self):
        '''
        Resets the environment to its initial state.
        Returns:  float -> Initial asset price
        '''
        self.asset_price = self.s0
        return self.asset_price
    def step(self):
        '''
        Advances the environment by one time step
        Returns: float -> updated asset price
        '''
        z = np.random.normal()
        self.asset_price *= np.exp((self.mu - 0.5 *self.sigma**2)*self.dt +
                                   self.sigma * np.sqrt(self.dt) * z)
        return self.asset_price
