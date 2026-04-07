"""
Market Simulator: Geometric Brownian Motion with optional extensions.

Supports three simulation modes:
  1. Standard GBM           — baseline, model vol == realized vol
  2. Volatility Mismatch    — realized_sigma != sigma (model misprices risk)
  3. Regime-Switching GBM   — two-state Markov-modulated volatility

The 'sigma' parameter always represents the MODEL volatility used for option
pricing. 'realized_sigma' (or regime vols) govern actual price dynamics.
This separation is the core of the vol-mismatch and regime-switching
experiments where RL learns to adapt but classical delta hedging cannot.
"""

import numpy as np


class Simulator:
    """
    GBM asset price simulator with optional regime switching.

    Args:
        s0:               Initial asset price
        mu:               Drift (annualized)
        sigma:            Model volatility for option pricing (annualized)
        dt:               Time step in years (e.g. 1/252 for daily)
        seed:             Random seed for reproducibility
        realized_sigma:   Actual vol for price simulation; if None uses sigma
        regime_switching: Enable two-state Markov-modulated vol
        sigma_low:        Low-regime annualized volatility
        sigma_high:       High-regime annualized volatility
        p_switch_per_year: Expected regime transitions per calendar year
    """

    def __init__(
        self,
        s0: float,
        mu: float,
        sigma: float,
        dt: float,
        seed: int | None = None,
        realized_sigma: float | None = None,
        regime_switching: bool = False,
        sigma_low: float = 0.15,
        sigma_high: float = 0.35,
        p_switch_per_year: float = 20.0,
    ):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma          # model vol (option pricing)
        self.dt = dt
        self.realized_sigma = realized_sigma if realized_sigma is not None else sigma
        self.regime_switching = regime_switching
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        # Convert annual switch rate to per-step probability
        # E.g. 20 switches/year * (1/252 year/step) ≈ 0.079 prob/step
        self.p_switch = min(p_switch_per_year * dt, 0.5)

        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> float:
        """Reset to initial price. Regime is randomised at episode start."""
        self.asset_price = self.s0
        # Start in random regime so the agent sees both during training
        self.regime = int(self.rng.integers(0, 2))
        return self.asset_price

    def step(self) -> tuple[float, float]:
        """
        Advance one time step via GBM.

        Returns:
            (new_price, realized_vol_this_step)
        """
        # Possibly switch regime
        if self.regime_switching and self.rng.random() < self.p_switch:
            self.regime = 1 - self.regime

        sigma_t = self._current_vol()
        z = self.rng.standard_normal()
        self.asset_price *= np.exp(
            (self.mu - 0.5 * sigma_t ** 2) * self.dt
            + sigma_t * np.sqrt(self.dt) * z
        )
        return self.asset_price, sigma_t

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_vol(self) -> float:
        if self.regime_switching:
            return self.sigma_low if self.regime == 0 else self.sigma_high
        return self.realized_sigma

    @property
    def current_regime(self) -> int:
        """0 = low-vol regime, 1 = high-vol regime."""
        return self.regime
