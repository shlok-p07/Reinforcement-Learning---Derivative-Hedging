"""
Hedging policies evaluated in this project.

Every policy exposes the same interface::

    action = policy(env, obs, prev_h)

where ``env`` is the live environment (read-only access to spot/tau/greeks),
``obs`` is the current observation, and ``prev_h`` is the holding carried into
this step.  Returning a target holding rather than a trade keeps the interface
identical to the RL action space, so learned and rule-based policies are
directly interchangeable in the evaluation harness.

Policies
--------
no_hedge         Hold zero stock (naked short call) — lower bound.
delta            Black-Scholes delta replication — the classical benchmark.
random           Uniform random target in [-1, 1] — sanity control.
whalley_wilmott  Asymptotically optimal no-transaction band under proportional
                 costs (Whalley & Wilmott, 1997).  The strongest classical
                 benchmark: unlike naive delta it is *cost-aware* by
                 construction, so beating it is a genuine result.

Whalley-Wilmott
---------------
For an agent with exponential utility (risk aversion ``gamma_risk``) facing
proportional transaction cost ``lambda``, the optimal policy is to do nothing
while the holding stays inside a band around the Black-Scholes delta, and to
trade only to the *nearest edge* of that band when it is breached::

    H = ( 3/2 · e^{-r·tau} · lambda · S · Gamma² / gamma_risk )^{1/3}

    no-trade region:  [ Delta - H , Delta + H ]

The band widens with transaction cost and with gamma (expensive, twitchy
hedges) and narrows as risk aversion rises.  ``H -> 0`` recovers continuous
delta hedging.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "no_hedge",
    "delta_hedge",
    "random_hedge",
    "make_whalley_wilmott",
    "RULE_BASED",
]


def no_hedge(env, obs, prev_h):  # noqa: ARG001
    """Hold nothing. Isolates the raw short-option exposure."""
    return 0.0


def delta_hedge(env, obs, prev_h):  # noqa: ARG001
    """Rebalance to Black-Scholes delta every step, ignoring cost.

    ``obs[2]`` is the option delta computed at the *current* (pre-step) state
    using the model volatility, so under a vol-misspecification scenario this
    is deliberately the wrong delta — which is the point of that test.
    """
    return float(obs[2])


def random_hedge(env, obs, prev_h):  # noqa: ARG001
    """Uniform random target holding — control for 'did the agent learn anything'."""
    return float(np.random.uniform(-1.0, 1.0))


def make_whalley_wilmott(gamma_risk: float = 1.0):
    """Build a Whalley-Wilmott band policy for a given risk aversion.

    Parameters
    ----------
    gamma_risk : float
        Exponential-utility risk-aversion coefficient.  Lower values tolerate
        more hedging error in exchange for lower turnover (wider band).

    Returns
    -------
    callable
        A policy with the standard ``(env, obs, prev_h) -> target_holding``
        signature.
    """
    if gamma_risk <= 0:
        raise ValueError("gamma_risk must be positive")

    def policy(env, obs, prev_h):  # noqa: ARG001
        tau = max(float(env.tau), 1e-8)
        spot = float(env.spot)
        delta = env.option.delta(spot, tau)
        gamma = env.option.gamma(spot, tau)
        lam = float(env.transaction_cost)

        # Degenerate cases: no cost, no convexity, or at expiry -> plain delta.
        if lam <= 0.0 or gamma <= 0.0:
            return float(delta)

        half_width = (
            1.5 * np.exp(-env.rate * tau) * lam * spot * gamma ** 2 / gamma_risk
        ) ** (1.0 / 3.0)

        lower, upper = delta - half_width, delta + half_width
        if prev_h < lower:
            return float(lower)   # trade up to the near edge, not all the way to delta
        if prev_h > upper:
            return float(upper)
        return float(prev_h)      # inside the band: do nothing, pay nothing

    policy.__name__ = f"whalley_wilmott_g{gamma_risk:g}"
    return policy


#: Cost-unaware rule-based policies, keyed by the name used in results tables.
RULE_BASED = {
    "no_hedge": no_hedge,
    "delta": delta_hedge,
    "random": random_hedge,
}
