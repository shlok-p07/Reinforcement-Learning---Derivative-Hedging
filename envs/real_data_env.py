"""
Real-market-data Gymnasium environment for option hedging.

Instead of simulating synthetic GBM price paths, this environment
replays actual SPY daily price windows from history.  Each episode
randomly samples a 30-day window, normalises prices to start at 100
so the agent learns from relative returns rather than absolute levels,
and uses that window's realised volatility for Black-Scholes pricing.

The agent therefore trains on genuine market dynamics:
  - Fat tails, skewness, leptokurtosis
  - Volatility clustering and GARCH-like persistence
  - Real market events (crashes, rallies, slow grinds)
  - Autocorrelated volatility regimes

Data requirements
-----------------
  data/spy_daily.csv must contain at minimum:
    date, close, rvol_21d

Run  python data/generate_data.py  to populate this file (5-year fetch).
"""

import os
import sys

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.black_scholes import EuropeanCallOption  # noqa: E402


class RealDataHedgingEnv(gym.Env):
    """
    Hedging environment driven by real SPY daily price history.

    Episode structure
    -----------------
    reset(): Sample a random 30-day window from the price history.
             Normalise all prices so the episode starts at S₀=100.
             Compute the BS implied vol from that window's realised vol.
    step():  Advance one trading day using the actual recorded price.
             Same reward and observation structure as RLHedgingEnv so
             models trained here are directly comparable.

    Parameters
    ----------
    data_path : str
        Absolute path to spy_daily.csv (must have 'close' and 'rvol_21d').
    window_size : int
        Episode length in trading days (default 30, matching a monthly option).
    strike_moneyness : float
        Strike as a fraction of the starting spot (1.0 = ATM, 1.05 = 5% OTM).
    rate : float
        Risk-free rate (annualised).
    transaction_cost : float
        Round-trip cost per unit of stock traded as a fraction of spot.
    augment_vol : bool
        If True, scale the window's realised vol by a random factor in
        [0.7, 1.3] to create additional volatility-regime variety during
        training.  Set False for pure historical backtest evaluation.
    seed : int | None
        RNG seed for reproducible episode sampling.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path: str,
        window_size: int = 30,
        strike_moneyness: float = 1.0,
        rate: float = 0.01,
        transaction_cost: float = 0.001,
        lambda_hedge: float = 1.0,
        lambda_terminal: float = 5.0,
        augment_vol: bool = True,
        seed: int | None = None,
    ):
        super().__init__()

        self.window_size       = window_size
        self.strike_moneyness  = strike_moneyness
        self.rate              = rate
        self.transaction_cost  = transaction_cost
        self.lambda_hedge      = lambda_hedge
        self.lambda_terminal   = lambda_terminal
        self.augment_vol       = augment_vol
        self.dt                = 1 / 252

        # ── Load and validate price history ──────────────────────────
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Price data not found at: {data_path}\n"
                "Run  python data/generate_data.py  to fetch SPY history."
            )

        df = pd.read_csv(data_path).sort_values("date").reset_index(drop=True)
        missing = {"close", "rvol_21d"} - set(df.columns)
        if missing:
            raise ValueError(f"spy_daily.csv is missing columns: {missing}")

        df = df.dropna(subset=["close", "rvol_21d"])
        self._prices = df["close"].values.astype(np.float64)
        self._rvols  = df["rvol_21d"].values.astype(np.float64)

        n = len(self._prices)
        if n < window_size + 1:
            raise ValueError(
                f"Need at least {window_size + 1} rows after dropping NaNs, "
                f"only {n} available.  Run generate_data.py to fetch more history."
            )

        # Valid episode starting indices
        self._starts = np.arange(0, n - window_size)
        self._rng    = np.random.default_rng(seed)

        # ── Spaces (identical to RLHedgingEnv for model compatibility) ─
        self.observation_space = spaces.Box(
            low =np.array([0.0, 0.0, 0.0, 0.0, -1.5, -1.0], dtype=np.float32),
            high=np.array([3.0, 1.0, 1.0,  2.0,  1.5,  1.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low =np.array([-1.5], dtype=np.float32),
            high=np.array([ 1.5], dtype=np.float32),
        )

        # Episode state — initialised here so type checkers are happy
        self._window_prices: np.ndarray = np.full(window_size + 1, 100.0)
        self._step_idx:  int   = 0
        self._start_idx: int   = 0
        self.s0:         float = 100.0
        self.spot:       float = 100.0
        self.strike:     float = 100.0
        self.sigma:      float = 0.20
        self.maturity:   float = window_size * self.dt
        self.tau:        float = window_size * self.dt
        self.stock_position: float = 0.0
        self.cash:           float = 0.0
        self.portfolio_value: float = 0.0
        self.total_tc:        float = 0.0
        self.option: EuropeanCallOption = EuropeanCallOption(
            self.strike, self.sigma, self.rate
        )

    # ── Gymnasium API ────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # ── Sample a random historical 30-day window ──────────────────
        self._start_idx = int(self._rng.choice(self._starts))
        raw = self._prices[self._start_idx : self._start_idx + self.window_size + 1]

        # Normalise: episode always starts at exactly 100
        self._window_prices = raw / raw[0] * 100.0
        self._step_idx = 0

        # Realised vol for this window — floor at 5%, cap at 80%
        raw_vol = float(
            np.mean(self._rvols[self._start_idx : self._start_idx + self.window_size])
        )
        self.sigma = float(np.clip(raw_vol, 0.05, 0.80))

        if self.augment_vol:
            # Scale by random factor to create additional vol-regime variety
            scale = float(self._rng.uniform(0.7, 1.3))
            self.sigma = float(np.clip(self.sigma * scale, 0.05, 0.80))

        # Episode parameters
        self.s0       = 100.0
        self.spot     = float(self._window_prices[0])  # = 100.0
        self.maturity = self.window_size * self.dt
        self.tau      = self.maturity
        self.strike   = self.s0 * self.strike_moneyness

        self.option           = EuropeanCallOption(self.strike, self.sigma, self.rate)
        self.stock_position   = 0.0
        self.total_tc         = 0.0
        self.cash             = self.option.price(self.spot, self.tau)
        self.portfolio_value  = 0.0          # V₀ = 0

        return self._obs(), {}

    def step(self, action):
        action = float(
            np.clip(np.asarray(action, dtype=float).flat[0], -1.5, 1.5)
        )
        prev_v = self.portfolio_value

        # ── Advance to next real historical price ─────────────────────
        self._step_idx += 1
        self.spot = float(self._window_prices[self._step_idx])
        self.tau  = max(self.tau - self.dt, 1e-8)

        # ── Rebalance ─────────────────────────────────────────────────
        trade   = action - self.stock_position
        tc      = abs(trade) * self.spot * self.transaction_cost
        self.cash          -= trade * self.spot + tc
        self.stock_position = action
        self.total_tc      += tc

        # ── Mark to market ────────────────────────────────────────────
        option_value         = self.option.price(self.spot, self.tau)
        self.portfolio_value = (
            self.stock_position * self.spot + self.cash - option_value
        )

        # ── Reward: penalise hedging-error variance, risk-averse ──────
        delta_v = self.portfolio_value - prev_v
        reward  = (
            -self.lambda_hedge * delta_v ** 2
            - 0.5 * self.lambda_hedge * max(-delta_v, 0.0) ** 2
        )

        terminated = self.tau <= 1e-8
        if terminated:
            reward -= self.lambda_terminal * self.portfolio_value ** 2

        info = {
            "portfolio_value": self.portfolio_value,
            "delta_v":         delta_v,
            "tc_step":         tc,
            "total_tc":        self.total_tc,
            "realized_vol":    self.sigma,
            "option_value":    option_value,
            "spot":            self.spot,
            "tau":             self.tau,
            "regime":          0,       # no synthetic regime in real data
        }
        return self._obs(), reward, terminated, False, info

    def render(self, mode="human"):  # noqa: ARG002
        pass

    # ── Internal ─────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        tau_safe = max(self.tau, 1e-8)
        delta    = self.option.delta(self.spot, tau_safe)
        gamma    = self.option.gamma(self.spot, tau_safe)
        obs = np.array([
            self.spot / self.s0,
            self.tau / self.maturity,
            delta,
            gamma * self.spot * np.sqrt(tau_safe),
            self.stock_position,
            np.log(self.spot / self.strike),
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ── Convenience ──────────────────────────────────────────────────

    @property
    def n_windows(self) -> int:
        """Total number of distinct 30-day windows available."""
        return len(self._starts)

    @property
    def current_window_start(self) -> int:
        """Index in the price array where the current episode began."""
        return self._start_idx
