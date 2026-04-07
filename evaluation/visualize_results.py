"""
Visualization pipeline — generates 6 publication-quality plots from evaluation results.

Plots produced (saved to results/plots/)
-----------------------------------------
  01_pnl_distributions.png   — KDE + histogram of terminal P&L, base scenario
  02_risk_return.png         — Mean P&L vs Std P&L scatter, all scenarios
  03_sharpe_comparison.png   — Sharpe ratio grouped bar chart across scenarios
  04_var_cvar_comparison.png — 95% VaR and CVaR bar chart across scenarios
  05_transaction_costs.png   — Average transaction costs by strategy × scenario
  06_learning_curves.png     — PPO and SAC reward curves during training

Run after full_evaluation.py and training scripts have completed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")   # headless backend — works on servers without a display
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from envs.rl_hedging_env import RLHedgingEnv  # noqa: E402

# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------

STRATEGY_COLORS = {
    "no_hedge": "#e74c3c",
    "random":   "#95a5a6",
    "delta":    "#3498db",
    "sac":      "#f39c12",
    "ppo":      "#2ecc71",
}

SCENARIO_LABELS = {
    "base":          "Base",
    "high_tc":       "High TC",
    "vol_mismatch":  "Vol Mismatch",
    "regime_switch": "Regime Switch",
}

PLOT_DIR = "results/plots"
RESULTS_CSV = "results/evaluation_results.csv"


def _save(fig, name: str):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------
# Plot 1 — P&L distributions, base scenario
# ------------------------------------------------------------------

def plot_pnl_distributions(df: pd.DataFrame, n_samples: int = 500):
    """KDE curves of terminal P&L for each strategy in the base scenario."""
    base = df[df["scenario"] == "base"]
    strategies = [s for s in ["no_hedge", "delta", "ppo", "sac"] if s in base["strategy"].values]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Re-simulate episodes to get raw P&L distributions for KDE
    env_kwargs = dict(
        s0=100.0, mu=0.05, sigma=0.20, dt=1 / 252,
        maturity=30 / 252, strike=100.0, rate=0.01,
        transaction_cost=0.001,
    )
    from stable_baselines3 import PPO, SAC  # noqa: PLC0415

    models = {}
    for name, cls in [("ppo", PPO), ("sac", SAC)]:
        path = os.path.join("models", f"{name}_hedger.zip")
        if os.path.exists(path):
            try:
                models[name] = cls.load(path.replace(".zip", ""))
            except Exception:  # noqa: BLE001
                pass

    for strategy in strategies:
        pnls = []
        for ep in range(n_samples):
            env = RLHedgingEnv(**env_kwargs, seed=ep)
            obs, _ = env.reset()
            done = False
            while not done:
                if strategy == "no_hedge":
                    action = np.array([0.0])
                elif strategy == "delta":
                    action = np.array([obs[2]])
                elif strategy in models:
                    action, _ = models[strategy].predict(obs, deterministic=True)
                else:
                    continue
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            pnls.append(env.portfolio_value)

        pnls = np.array(pnls)
        color = STRATEGY_COLORS.get(strategy, "#333")
        kde = gaussian_kde(pnls, bw_method=0.3)
        xs = np.linspace(pnls.min() - 0.5, pnls.max() + 0.5, 300)
        ax.plot(xs, kde(xs), color=color, lw=2.5, label=strategy.upper())
        ax.axvline(np.percentile(pnls, 5), color=color, lw=1.2, ls="--", alpha=0.7)

    ax.axvline(0, color="black", lw=1, ls=":", alpha=0.5)
    ax.set_xlabel("Terminal Portfolio P&L", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("P&L Distributions — Base Scenario\n(dashed = 95% VaR)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save(fig, "01_pnl_distributions.png")


# ------------------------------------------------------------------
# Plot 2 — Risk-return scatter
# ------------------------------------------------------------------

def plot_risk_return(df: pd.DataFrame):
    """Mean P&L vs Std P&L for all strategy × scenario combinations."""
    fig, ax = plt.subplots(figsize=(9, 7))

    scenario_markers = {"base": "o", "high_tc": "s", "vol_mismatch": "D", "regime_switch": "^"}

    for _, row in df.iterrows():
        color = STRATEGY_COLORS.get(row["strategy"], "#333")
        marker = scenario_markers.get(row["scenario"], "o")
        ax.scatter(
            row["std_pnl"], row["mean_pnl"],
            color=color, marker=marker, s=120, zorder=3,
            edgecolors="white", linewidths=0.8,
        )
        ax.annotate(
            f"{row['strategy'][:3]}", (row["std_pnl"], row["mean_pnl"]),
            fontsize=7, ha="left", va="bottom",
            xytext=(3, 3), textcoords="offset points",
        )

    # Legend patches
    strategy_patches = [
        mpatches.Patch(color=c, label=s.upper())
        for s, c in STRATEGY_COLORS.items() if s in df["strategy"].values
    ]
    scenario_handles = [
        plt.Line2D([0], [0], marker=m, color="grey", ls="", markersize=9,
                   label=SCENARIO_LABELS.get(sc, sc))
        for sc, m in scenario_markers.items()
    ]
    ax.legend(
        handles=strategy_patches + scenario_handles,
        loc="upper right", fontsize=9, ncol=2,
    )
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("P&L Standard Deviation  (risk)", fontsize=13)
    ax.set_ylabel("Mean Terminal P&L  (return)", fontsize=13)
    ax.set_title("Risk-Return Landscape\nAll strategies × all scenarios", fontsize=14)
    ax.grid(alpha=0.3)
    _save(fig, "02_risk_return.png")


# ------------------------------------------------------------------
# Plot 3 — Sharpe ratio comparison
# ------------------------------------------------------------------

def plot_sharpe(df: pd.DataFrame):
    """Grouped bar chart: Sharpe ratio per strategy, grouped by scenario."""
    scenarios = list(SCENARIO_LABELS.keys())
    strategies = [s for s in STRATEGY_COLORS if s in df["strategy"].values]
    n_scenarios = len(scenarios)
    n_strats = len(strategies)
    width = 0.8 / n_strats
    xs = np.arange(n_scenarios)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, strategy in enumerate(strategies):
        sharpes = []
        for sc in scenarios:
            row = df[(df["strategy"] == strategy) & (df["scenario"] == sc)]
            sharpes.append(float(row["sharpe"].values[0]) if len(row) > 0 else 0.0)
        offset = (i - n_strats / 2 + 0.5) * width
        bars = ax.bar(xs + offset, sharpes, width * 0.9,
                      color=STRATEGY_COLORS[strategy], label=strategy.upper(),
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, sharpes):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5,
            )

    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=12)
    ax.set_ylabel("Sharpe Ratio  (mean P&L / std P&L)", fontsize=12)
    ax.set_title("Sharpe Ratio by Strategy and Scenario\n(higher = better risk-adjusted return)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "03_sharpe_comparison.png")


# ------------------------------------------------------------------
# Plot 4 — VaR / CVaR comparison
# ------------------------------------------------------------------

def plot_var_cvar(df: pd.DataFrame):
    """Side-by-side VaR-95 and CVaR-95 bars for each scenario."""
    scenarios = list(SCENARIO_LABELS.keys())
    strategies = [s for s in STRATEGY_COLORS if s in df["strategy"].values]
    n_strats = len(strategies)
    width = 0.8 / n_strats
    xs = np.arange(len(scenarios))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, metric, title in zip(
        axes,
        ["var_95", "cvar_95"],
        ["95% VaR (5th percentile P&L)", "95% CVaR (Expected Shortfall)"],
    ):
        for i, strategy in enumerate(strategies):
            vals = []
            for sc in scenarios:
                row = df[(df["strategy"] == strategy) & (df["scenario"] == sc)]
                vals.append(float(row[metric].values[0]) if len(row) > 0 else 0.0)
            offset = (i - n_strats / 2 + 0.5) * width
            ax.bar(xs + offset, vals, width * 0.9,
                   color=STRATEGY_COLORS[strategy], label=strategy.upper(),
                   edgecolor="white", linewidth=0.5)

        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("P&L (negative = loss)", fontsize=12)
            ax.legend(fontsize=10)

    fig.suptitle("Tail Risk Comparison — Higher (less negative) is better", fontsize=14, y=1.02)
    _save(fig, "04_var_cvar_comparison.png")


# ------------------------------------------------------------------
# Plot 5 — Transaction cost breakdown
# ------------------------------------------------------------------

def plot_transaction_costs(df: pd.DataFrame):
    """Average transaction costs paid per episode, by strategy × scenario."""
    scenarios = list(SCENARIO_LABELS.keys())
    strategies = [s for s in STRATEGY_COLORS if s in df["strategy"].values]
    n_strats = len(strategies)
    width = 0.8 / n_strats
    xs = np.arange(len(scenarios))

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, strategy in enumerate(strategies):
        tcs = []
        for sc in scenarios:
            row = df[(df["strategy"] == strategy) & (df["scenario"] == sc)]
            tcs.append(float(row["avg_tc"].values[0]) if len(row) > 0 else 0.0)
        offset = (i - n_strats / 2 + 0.5) * width
        bars = ax.bar(xs + offset, tcs, width * 0.9,
                      color=STRATEGY_COLORS[strategy], label=strategy.upper(),
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, tcs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=12)
    ax.set_ylabel("Average Total Transaction Cost per Episode", fontsize=12)
    ax.set_title("Transaction Costs by Strategy and Scenario\n(RL learns to minimise unnecessary rebalancing)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "05_transaction_costs.png")


# ------------------------------------------------------------------
# Plot 6 — Learning curves
# ------------------------------------------------------------------

def plot_learning_curves():
    """PPO and SAC mean eval reward vs training timesteps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    found_any = False

    for name, color in [("ppo", "#2ecc71"), ("sac", "#f39c12")]:
        npz_path = os.path.join("results", "learning_curves", name, "evaluations.npz")
        if not os.path.exists(npz_path):
            print(f"  Note: {npz_path} not found (run training first)")
            continue
        found_any = True
        data = np.load(npz_path)
        timesteps = data["timesteps"]
        results = data["results"]          # shape: (n_evals, n_episodes)
        means = results.mean(axis=1)
        stds = results.std(axis=1)
        ax.plot(timesteps, means, color=color, lw=2.5, label=name.upper())
        ax.fill_between(
            timesteps, means - stds, means + stds,
            color=color, alpha=0.2,
        )

    if not found_any:
        ax.text(0.5, 0.5, "Train agents first:\n  python training/train_ppo.py\n  python training/train_sac.py",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round", facecolor="lightyellow"))

    ax.set_xlabel("Training Timesteps", fontsize=13)
    ax.set_ylabel("Mean Episode Reward  (± 1 std)", fontsize=13)
    ax.set_title("Learning Curves — PPO and SAC\n(upward trend = agent improving hedge quality)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    _save(fig, "06_learning_curves.png")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating plots...")

    if not os.path.exists(RESULTS_CSV):
        print(f"ERROR: {RESULTS_CSV} not found. Run evaluation/full_evaluation.py first.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")

    plot_pnl_distributions(df)
    plot_risk_return(df)
    plot_sharpe(df)
    plot_var_cvar(df)
    plot_transaction_costs(df)
    plot_learning_curves()

    print(f"\nAll plots saved to {PLOT_DIR}/")
