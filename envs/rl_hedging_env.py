"""RL environment for European call option hedging."""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Allow imports from project root regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.market_simulator import Simulator  # noqa: E402
from utils.black_scholes import EuropeanCallOption  # noqa: E402


class RLHedgingEnv(gym.Env):
    """
    Gymnasium environment for dynamic option hedging.

    Financial Setup
    ---------------
    The agent has SOLD a European call and must hedge the short-gamma exposure.
    At t=0 it receives the Black-Scholes premium as cash. Each step it picks a
    target stock holding h_t ∈ [-1.5, 1.5]. At expiry the option settles at
    max(S_T - K, 0).

    Portfolio:  V_t = h_t * S_t + cash_t - C(S_t, τ_t)
    V_0 = 0  (premium exactly offsets liability under BS assumptions).

    Observation (6 features, normalised)
    ------------------------------------
      [0] spot / s0          normalised price (~1.0 at par)
      [1] tau / maturity     fraction of time remaining ∈ [0, 1]
      [2] delta              option delta N(d1) ∈ [0, 1]
      [3] gamma_norm         gamma * spot * sqrt(tau), dimensionless
      [4] stock_position     current holding ∈ [-1.5, 1.5]
      [5] log_moneyness      log(spot / strike)

    Reward
    ------
      Step:     r_t = -λ_h * (ΔV_t)²  − 0.5 λ_h * max(-ΔV_t, 0)²
      Terminal: r_T += -λ_T * V_T²

    ΔV_t is the one-step portfolio change (hedging error). The extra downside
    term makes the agent risk-averse. λ_T amplifies the final settlement signal.
    """

    metadata = {"render_modes": []}

    def __init__(  # pylint: disable=too-many-arguments
        self,
        s0: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.20,
        dt: float = 1 / 252,
        maturity: float = 30 / 252,
        strike: float = 100.0,
        rate: float = 0.01,
        transaction_cost: float = 0.001,
        realized_sigma: float | None = None,
        regime_switching: bool = False,
        sigma_low: float = 0.15,
        sigma_high: float = 0.35,
        lambda_hedge: float = 1.0,
        lambda_terminal: float = 5.0,
        seed: int | None = None,
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
        self.lambda_hedge = lambda_hedge
        self.lambda_terminal = lambda_terminal

        self.simulator = Simulator(
            s0=s0, mu=mu, sigma=sigma, dt=dt, seed=seed,
            realized_sigma=realized_sigma,
            regime_switching=regime_switching,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
        )
        # Option pricing uses MODEL sigma — the intentional mismatch RL must learn to handle
        self.option = EuropeanCallOption(strike, sigma, rate)

        # Observation: [spot_norm, time_frac, delta, gamma_norm, position, log_moneyness]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.5, -1.0], dtype=np.float32),
            high=np.array([3.0, 1.0, 1.0, 2.0,  1.5,  1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.5], dtype=np.float32),
            high=np.array([ 1.5], dtype=np.float32),
            dtype=np.float32,
        )

        self.tau: float = maturity
        self.spot: float = s0
        self.stock_position: float = 0.0
        self.cash: float = 0.0
        self.portfolio_value: float = 0.0
        self.total_tc: float = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.simulator = Simulator(
                s0=self.s0, mu=self.mu, sigma=self.sigma, dt=self.dt,
                seed=seed,
                realized_sigma=self.simulator.realized_sigma,
                regime_switching=self.simulator.regime_switching,
                sigma_low=self.simulator.sigma_low,
                sigma_high=self.simulator.sigma_high,
            )
        self.tau = self.maturity
        self.stock_position = 0.0
        self.total_tc = 0.0
        self.spot = self.simulator.reset()
        self.cash = self.option.price(self.spot, self.tau)   # receive premium
        self.portfolio_value = 0.0                            # V_0 = 0
        return self._obs(), {}

    def step(self, action):
        # Accept scalar, 0-d array, or 1-d array of length 1
        action = float(np.clip(np.asarray(action, dtype=float).flat[0], -1.5, 1.5))
        prev_v = self.portfolio_value

        self.spot, realized_vol = self.simulator.step()
        self.tau = max(self.tau - self.dt, 1e-8)

        trade = action - self.stock_position
        tc = abs(trade) * self.spot * self.transaction_cost
        self.cash -= trade * self.spot + tc
        self.stock_position = action
        self.total_tc += tc

        option_value = self.option.price(self.spot, self.tau)
        self.portfolio_value = (
            self.stock_position * self.spot + self.cash - option_value
        )

        delta_v = self.portfolio_value - prev_v
        reward = (
            -self.lambda_hedge * delta_v ** 2
            - 0.5 * self.lambda_hedge * max(-delta_v, 0.0) ** 2
        )

        terminated = self.tau <= 1e-8
        if terminated:
            reward -= self.lambda_terminal * self.portfolio_value ** 2

        info = {
            "portfolio_value": self.portfolio_value,
            "delta_v": delta_v,
            "tc_step": tc,
            "total_tc": self.total_tc,
            "realized_vol": realized_vol,
            "option_value": option_value,
            "spot": self.spot,
            "tau": self.tau,
            "regime": self.simulator.current_regime,
        }
        return self._obs(), reward, terminated, False, info

    def render(self, mode="human"):  # noqa: ARG002
        pass

    def _obs(self) -> np.ndarray:
        tau_safe = max(self.tau, 1e-8)
        delta = self.option.delta(self.spot, tau_safe)
        gamma = self.option.gamma(self.spot, tau_safe)
        obs = np.array([
            self.spot / self.s0,
            self.tau / self.maturity,
            delta,
            gamma * self.spot * np.sqrt(tau_safe),
            self.stock_position,
            np.log(self.spot / self.strike),
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
