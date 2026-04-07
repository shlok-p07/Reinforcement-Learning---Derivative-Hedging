"""
Comprehensive evaluation: 5 strategies × 4 market scenarios × 8 metrics.

Strategies
----------
  no_hedge    — hold zero stock throughout (naked short option)
  delta       — replicate via Black-Scholes delta at each step
  random      — uniform random action ∈ [-1, 1] (sanity check)
  ppo         — trained PPO agent (deterministic rollout)
  sac         — trained SAC agent (deterministic rollout)

Scenarios
---------
  base         — calibrated market (σ_model = σ_realized = 20 %, TC = 0.1 %)
  high_tc      — 10× transaction costs (TC = 1 %)  → tests rebalancing discipline
  vol_mismatch — model assumes 20 %, market realises 30 %  → tests adaptability
  regime_switch— vol alternates between 15 % and 35 %  → tests regime detection

Metrics per strategy × scenario
--------------------------------
  mean_pnl     — average terminal portfolio value
  std_pnl      — standard deviation of terminal P&L
  sharpe       — mean_pnl / std_pnl (risk-adjusted return)
  var_95       — 5th percentile of P&L (95 % VaR, sign convention: losses negative)
  cvar_95      — expected P&L below VaR (Conditional VaR / Expected Shortfall)
  max_loss     — worst single episode
  pct_loss     — fraction of episodes ending with negative P&L
  avg_tc       — average total transaction cost paid per episode

Results are saved to results/evaluation_results.csv.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC  # noqa: E402

from envs.rl_hedging_env import RLHedgingEnv  # noqa: E402

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

N_EPISODES = 500
RESULTS_DIR = "results"
MODEL_DIR = "models"

BASE_ENV = dict(
    s0=100.0, mu=0.05, sigma=0.20, dt=1 / 252,
    maturity=30 / 252, strike=100.0, rate=0.01,
    transaction_cost=0.001,
)

SCENARIOS: dict[str, dict] = {
    "base": BASE_ENV,
    "high_tc": {**BASE_ENV, "transaction_cost": 0.01},
    "vol_mismatch": {**BASE_ENV, "realized_sigma": 0.30},
    "regime_switch": {
        **BASE_ENV,
        "regime_switching": True,
        "sigma_low": 0.15,
        "sigma_high": 0.35,
    },
}

# ------------------------------------------------------------------
# Episode runners
# ------------------------------------------------------------------

def run_episode_baseline(env: RLHedgingEnv, strategy: str) -> dict:
    """Run one episode with a rule-based strategy, return metrics dict."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if strategy == "no_hedge":
            action = np.array([0.0], dtype=np.float32)
        elif strategy == "delta":
            # obs[2] is the option delta (already in [0,1])
            action = np.array([obs[2]], dtype=np.float32)
        elif strategy == "random":
            action = np.array(
                [np.random.uniform(-1.0, 1.0)], dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    return {
        "pnl": env.portfolio_value,
        "total_tc": env.total_tc,
        "total_reward": total_reward,
    }


def run_episode_rl(env: RLHedgingEnv, model) -> dict:
    """Run one episode with a trained RL model (deterministic)."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    return {
        "pnl": env.portfolio_value,
        "total_tc": env.total_tc,
        "total_reward": total_reward,
    }


# ------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    """Aggregate episode results into the 8 evaluation metrics."""
    pnls = np.array([r["pnl"] for r in results])
    tcs = np.array([r["total_tc"] for r in results])

    var_95 = float(np.percentile(pnls, 5))
    tail = pnls[pnls <= var_95]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95
    std = float(np.std(pnls))

    return {
        "mean_pnl":  float(np.mean(pnls)),
        "std_pnl":   std,
        "sharpe":    float(np.mean(pnls) / std) if std > 1e-9 else 0.0,
        "var_95":    var_95,
        "cvar_95":   cvar_95,
        "max_loss":  float(np.min(pnls)),
        "pct_loss":  float(np.mean(pnls < 0)),
        "avg_tc":    float(np.mean(tcs)),
    }


# ------------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------------

def evaluate_all(n_episodes: int = N_EPISODES) -> pd.DataFrame:
    """Run all strategy × scenario combinations. Returns results DataFrame."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load RL models (skip gracefully if not trained yet)
    models: dict[str, object | None] = {"ppo": None, "sac": None}
    for name, cls in [("ppo", PPO), ("sac", SAC)]:
        path = os.path.join(MODEL_DIR, f"{name}_hedger")
        if os.path.exists(f"{path}.zip"):
            try:
                models[name] = cls.load(path)
                print(f"Loaded {name.upper()} from {path}.zip")
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: could not load {name}: {exc}")
        else:
            print(f"Warning: {path}.zip not found — skipping {name.upper()}")

    rows = []

    for scenario_name, env_kwargs in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}  ({n_episodes} episodes each)")
        print(f"{'='*60}")

        strategies = ["no_hedge", "delta", "random"]
        if models["ppo"] is not None:
            strategies.append("ppo")
        if models["sac"] is not None:
            strategies.append("sac")

        for strategy in strategies:
            episode_results = []

            for ep in range(n_episodes):
                env = RLHedgingEnv(**env_kwargs, seed=ep)

                if strategy in ("no_hedge", "delta", "random"):
                    ep_data = run_episode_baseline(env, strategy)
                elif strategy == "ppo":
                    ep_data = run_episode_rl(env, models["ppo"])
                else:
                    ep_data = run_episode_rl(env, models["sac"])

                episode_results.append(ep_data)

            metrics = compute_metrics(episode_results)

            row = {"strategy": strategy, "scenario": scenario_name, **metrics}
            rows.append(row)

            print(
                f"  {strategy:12s} | "
                f"mean={metrics['mean_pnl']:+.4f} "
                f"std={metrics['std_pnl']:.4f} "
                f"sharpe={metrics['sharpe']:+.3f} "
                f"VaR95={metrics['var_95']:+.4f} "
                f"CVaR95={metrics['cvar_95']:+.4f} "
                f"tc={metrics['avg_tc']:.4f}"
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Also save as JSON for easy programmatic access
    json_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    df.to_json(json_path, orient="records", indent=2)

    return df


if __name__ == "__main__":
    df = evaluate_all()
    print("\n\nFull Results Table:")
    print(df.to_string(index=False))
