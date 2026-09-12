"""
Paired bootstrap confidence intervals for hedging-strategy comparisons.

Why paired
----------
The evaluation harness runs every strategy on the *same* seed for a given
episode index, so strategy A and strategy B see an identical price path in
episode i.  This is the method of common random numbers: it removes path
noise from the comparison, and the difference ``pnl_A[i] - pnl_B[i]`` is a
genuinely paired observation.

Resampling therefore has to resample *episode indices*, applying the same
index set to both strategies.  Resampling each strategy independently would
throw away the pairing and inflate the interval.

What is reported
----------------
For each metric and each (strategy, benchmark) pair:

  * the observed difference  ``metric(A) - metric(B)``
  * a percentile bootstrap 95 % CI for that difference
  * a two-sided bootstrap p-value

The p-value is the achieved significance level of the bootstrap distribution:
twice the smaller tail mass on either side of zero.  A CI that excludes zero
and a p-value below 0.05 are the same statement.

Note on metrics: all metrics are computed on *terminal P&L per episode*, so a
"difference in Sharpe" is a difference of two ratios, not a ratio of
differences.  Nonlinear functionals like VaR and CVaR are exactly why the
bootstrap is used here rather than a closed-form standard error.
"""

from __future__ import annotations

import numpy as np

__all__ = ["METRICS", "paired_bootstrap", "compare_all"]


# ----------------------------------------------------------------------
# Vectorised metrics: each maps an array of shape (..., n_episodes)
# to shape (...), so they work on a single sample and on a whole
# (n_boot, n_episodes) resample matrix with the same code path.
# ----------------------------------------------------------------------

def _mean(x):
    return x.mean(axis=-1)


def _std(x):
    return x.std(axis=-1)


def _sharpe(x):
    s = x.std(axis=-1)
    return np.divide(x.mean(axis=-1), s, out=np.zeros_like(s), where=s > 1e-9)


def _var95(x):
    return np.percentile(x, 5, axis=-1)


def _cvar95(x):
    """Mean P&L conditional on breaching the 5th percentile (expected shortfall)."""
    thresh = np.percentile(x, 5, axis=-1, keepdims=True)
    mask = x <= thresh
    count = mask.sum(axis=-1)
    total = np.where(mask, x, 0.0).sum(axis=-1)
    return np.divide(total, count, out=np.squeeze(thresh, -1).copy(), where=count > 0)


def _max_loss(x):
    return x.min(axis=-1)


def _pct_loss(x):
    return (x < 0).mean(axis=-1)


#: Metric name -> vectorised functional. Higher is better for every entry
#: except ``pct_loss`` (and note all P&L metrics are costs here, so "higher"
#: means "less negative").
METRICS = {
    "mean_pnl": _mean,
    "std_pnl": _std,
    "sharpe": _sharpe,
    "var_95": _var95,
    "cvar_95": _cvar95,
    "max_loss": _max_loss,
    "pct_loss": _pct_loss,
}


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    metric: str,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap the difference ``metric(a) - metric(b)`` under pairing.

    Parameters
    ----------
    a, b : np.ndarray
        Per-episode terminal P&L for the two strategies, aligned so that
        ``a[i]`` and ``b[i]`` come from the same seed / price path.
    metric : str
        Key into :data:`METRICS`.
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Two-sided significance level; 0.05 gives a 95 % interval.
    seed : int
        RNG seed, so reported intervals are reproducible.

    Returns
    -------
    dict with keys ``metric, value_a, value_b, diff, ci_low, ci_high,
    p_value, significant``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")

    fn = METRICS[metric]
    n = a.shape[0]
    rng = np.random.default_rng(seed)

    # One index matrix, applied to BOTH strategies -> pairing preserved.
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = fn(a[idx]) - fn(b[idx])

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided achieved significance level.
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = float(min(p, 1.0))

    return {
        "metric": metric,
        "value_a": float(fn(a)),
        "value_b": float(fn(b)),
        "diff": float(fn(a) - fn(b)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p,
        "significant": bool(lo > 0 or hi < 0),
    }


def compare_all(
    pnl_by_strategy: dict[str, np.ndarray],
    benchmark: str,
    metrics=None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[dict]:
    """Bootstrap every strategy against ``benchmark`` across every metric."""
    metrics = list(METRICS) if metrics is None else list(metrics)
    base = pnl_by_strategy[benchmark]

    rows = []
    for name, pnl in pnl_by_strategy.items():
        if name == benchmark:
            continue
        for m in metrics:
            row = paired_bootstrap(pnl, base, m, n_boot=n_boot, seed=seed)
            row.update(strategy=name, benchmark=benchmark)
            rows.append(row)
    return rows
