"""
Reward objectives for the hedging environments.

Two objectives are available, selected by the ``risk_objective`` argument on
either environment.

``"quadratic"`` (default, unchanged behaviour)
----------------------------------------------
    r_t = -lam_h * dV_t^2 - 0.5 * lam_h * max(-dV_t, 0)^2
    r_T += -lam_T * V_T^2

A mean-squared-error objective with a mild downside asymmetry.  It is
minimised by the policy that makes hedging error *small on average*.  Because
squared error is an expectation, this objective is deliberately indifferent to
the shape of the loss tail: two policies with identical mean squared error
score identically even if one of them occasionally loses five times as much.
That is the documented reason the PPO agent trained under this objective shows
a wider 95 % CVaR than delta hedging.

``"cvar"`` (tail-sensitive alternative)
---------------------------------------
Adds the Rockafellar-Uryasev representation of Conditional Value at Risk::

    CVaR_q(V) = min over nu of  { nu + (1/q) * E[ (nu - V)^+ ] }

The inner expectation is an ordinary per-episode quantity, so it can be added
straight onto the terminal reward::

    r_T += -lam_c * (1/q) * max(nu - V_T, 0)

which charges the policy only for terminal outcomes that fall below ``nu``, in
proportion to how far below they fall.  ``nu`` is the Value at Risk itself and
is not known in advance, so it is tracked online by the standard stochastic
quantile approximation::

    nu <- nu + lr * ( q - 1{V_T < nu} )

whose fixed point is the q-quantile of the terminal P&L distribution: the
update pushes ``nu`` down by ``lr*(1-q)`` whenever an episode lands in the tail
and up by ``lr*q`` otherwise, so it balances exactly when a fraction ``q`` of
episodes fall below it.

Reference
---------
Rockafellar & Uryasev (2000), "Optimization of Conditional Value-at-Risk",
Journal of Risk 2(3), 21-41.
"""

from __future__ import annotations

__all__ = ["RiskObjective"]


class RiskObjective:
    """Computes step and terminal rewards under the selected objective.

    Parameters
    ----------
    kind : {"quadratic", "cvar"}
        Which objective to apply.
    lambda_hedge, lambda_terminal : float
        Weights on the per-step squared hedging error and the terminal
        settlement penalty.  Shared by both objectives.
    cvar_alpha : float
        Confidence level; ``0.95`` targets the worst 5 % of episodes.
    cvar_weight : float
        Weight ``lam_c`` on the CVaR term relative to the squared-error terms.
    cvar_lr : float
        Step size for the online VaR quantile tracker.
    var_init : float
        Starting guess for ``nu``.  The tracker is robust to this; it only
        affects the first few hundred episodes.

    Notes
    -----
    The VaR estimate is per-environment-instance state.  Under a vectorised
    trainer each worker maintains its own tracker, which is fine: they all
    converge to the same quantile of the same distribution and the resulting
    reward noise is no worse than the usual sampling noise.
    """

    def __init__(
        self,
        kind: str = "quadratic",
        lambda_hedge: float = 1.0,
        lambda_terminal: float = 5.0,
        cvar_alpha: float = 0.95,
        cvar_weight: float = 1.0,
        cvar_lr: float = 0.01,
        var_init: float = 0.0,
    ):
        if kind not in ("quadratic", "cvar"):
            raise ValueError(f"risk_objective must be 'quadratic' or 'cvar', got {kind!r}")
        if not 0.0 < cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie strictly between 0 and 1")

        self.kind = kind
        self.lambda_hedge = lambda_hedge
        self.lambda_terminal = lambda_terminal
        self.q = 1.0 - cvar_alpha          # tail probability, e.g. 0.05
        self.cvar_weight = cvar_weight
        self.cvar_lr = cvar_lr
        self.var_estimate = var_init

    def step_reward(self, delta_v: float) -> float:
        """Per-step reward. Identical under both objectives."""
        return (
            -self.lambda_hedge * delta_v ** 2
            - 0.5 * self.lambda_hedge * max(-delta_v, 0.0) ** 2
        )

    def terminal_reward(self, terminal_value: float) -> float:
        """Terminal reward, and (under ``cvar``) advance the VaR tracker."""
        reward = -self.lambda_terminal * terminal_value ** 2

        if self.kind == "cvar":
            shortfall = max(self.var_estimate - terminal_value, 0.0)
            reward -= self.cvar_weight * shortfall / self.q
            # Online q-quantile update; fixed point is the true VaR.
            indicator = 1.0 if terminal_value < self.var_estimate else 0.0
            self.var_estimate += self.cvar_lr * (self.q - indicator)

        return reward

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.kind == "quadratic":
            return f"RiskObjective(quadratic, lam_h={self.lambda_hedge}, lam_T={self.lambda_terminal})"
        return (
            f"RiskObjective(cvar, q={self.q:g}, w={self.cvar_weight}, "
            f"lr={self.cvar_lr}, VaR~{self.var_estimate:.3f})"
        )
