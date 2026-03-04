import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

from utils.market_simulator import Simulator
from utils.black_scholes import EuropeanCallOption


class RLHedgingEnv(gym.Env):
    """
    Reinforcement Learning Environment for Option Hedging.

    Observation:
        [spot, tau, delta, gamma, stock_position]

    Action:
        Continuous hedge ratio in range [-2, 2]

    Reward:
        Negative squared hedging error (portfolio value)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        s0: float,
        mu: float,
        sigma: float,
        dt: float,
        maturity: float,
        strike: float,
        rate: float,
        transaction_cost: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.maturity = maturity
        self.strike = strike
        self.rate = rate
        self.transaction_cost = transaction_cost
        self.seed_value = seed

        self.simulator = Simulator(s0, mu, sigma, dt, seed=seed)
        self.option = EuropeanCallOption(strike, sigma, rate)

        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0, -2.0], dtype=np.float32),
            high=np.array([500.0, 1.0, 1.0, 5.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Action space (hedge ratio)
        self.action_space = spaces.Box(
            low=np.array([-2.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset(seed=seed)

    # ============================================================
    # Reset
    # ============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.simulator = Simulator(
                self.s0, self.mu, self.sigma, self.dt, seed=seed
            )

        self.tau = self.maturity
        self.spot = self.simulator.reset()

        self.stock_position = 0.0
        self.cash = 0.0

        state = self._get_state()
        return state, {}

    # ============================================================
    # Step
    # ============================================================

    def step(self, action):
        # PPO sends action as array
        action = float(action[0])

        # Clip action to bounds
        action = np.clip(action, -2.0, 2.0)

        # 1. Market moves
        self.spot = self.simulator.step()
        self.tau -= self.dt
        self.tau = max(self.tau, 1e-8)

        # 2. Adjust hedge
        trade = action - self.stock_position
        self.cash -= trade * self.spot
        self.cash -= abs(trade) * self.spot * self.transaction_cost
        self.stock_position = action

        # 3. Compute option value
        option_price = self.option.price(self.spot, self.tau)

        # 4. Portfolio value
        portfolio_value = (
            self.stock_position * self.spot
            + self.cash
            - option_price
        )

        # 5. Reward (risk penalty)
        reward = - float(portfolio_value ** 2)

        # 6. Termination condition
        terminated = self.tau <= 1e-8
        truncated = False

        state = self._get_state()

        return state, reward, terminated, truncated, {}

    # ============================================================
    # State
    # ============================================================

    def _get_state(self):
        delta = self.option.delta(self.spot, self.tau)
        gamma = self.option.gamma(self.spot, self.tau)

        return np.array(
            [
                self.spot,
                self.tau,
                delta,
                gamma,
                self.stock_position,
            ],
            dtype=np.float32,
        )