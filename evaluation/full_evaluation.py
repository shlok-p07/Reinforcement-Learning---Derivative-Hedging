"""
Comprehensive evaluation: 6 strategies x 4 market scenarios x 8 metrics,
with paired-bootstrap confidence intervals on every comparison.

Strategies
----------
  no_hedge         hold zero stock throughout (naked short call)
  delta            Black-Scholes delta replication at every step
  whalley_wilmott  cost-aware no-transaction band (Whalley & Wilmott, 1997)
  random           uniform random action in [-1, 1] (sanity control)
  ppo              trained PPO agent (deterministic rollout)
  sac              trained SAC agent (deterministic rollout)

Scenarios
---------
  base          calibrated market (sigma_model = sigma_realized = 20 %, TC = 0.1 %)
  high_tc       10x transaction costs (TC = 1 %)      -> tests rebalancing discipline
  vol_mismatch  model assumes 20 %, market realises 30 % -> tests adaptability
  regime_switch vol alternates between 15 % and 35 %  -> tests regime detection

Methodology
-----------
Common random numbers: episode i uses ``seed=i`` for *every* strategy, so all
strategies are compared on identical price paths.  This is a variance-reduction
technique -- it removes path noise from strategy differences and makes the
per-episode P&L differences genuinely paired, which is what licenses the paired
bootstrap in ``evaluation.bootstrap``.

The Whalley-Wilmott risk-aversion parameter is calibrated per scenario on a
*disjoint* held-out seed block (``WW_TUNE_SEEDS``) and then frozen for the
reported evaluation seeds.  Tuning it on the evaluation seeds would flatter the
benchmark; using an untuned value would strawman it.

Outputs
-------
  results/evaluation_results.csv   aggregate metrics per strategy x scenario
  results/evaluation_results.json  same, JSON
  results/episode_pnl.csv          per-episode terminal P&L (tidy, for re-analysis)
  results/significance.csv         bootstrap CIs and p-values vs the delta benchmark
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402

from envs.rl_hedging_env import RLHedgingEnv  # noqa: E402
from evaluation.bootstrap import compare_all  # noqa: E402
from evaluation.hedging_strategies import RULE_BASED, make_whalley_wilmott  # noqa: E402

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

N_EPISODES = 450
N_BOOT = 10_000
BENCHMARK = "delta"          # every strategy is tested against this
RESULTS_DIR = "results"
MODEL_DIR = "models"

# Disjoint from range(N_EPISODES) so WW calibration is genuinely out-of-sample.
WW_TUNE_SEEDS = range(10_000, 10_200)
WW_GAMMA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

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
# Episode runner
# ------------------------------------------------------------------

def run_episode(env_kwargs: dict, policy, seed: int) -> dict:
    """Roll out one episode under ``policy``. Returns terminal P&L and costs.

    ``policy`` has the uniform signature ``(env, obs, prev_h) -> target_holding``
    so learned and rule-based strategies share this code path exactly.
    """
    env = RLHedgingEnv(**env_kwargs, seed=seed)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    prev_h = 0.0
    n_trades = 0

    while not done:
        action = policy(env, obs, prev_h)
        action = float(np.asarray(action, dtype=float).flat[0])
        if abs(action - prev_h) > 1e-9:
            n_trades += 1
        obs, reward, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
        prev_h = env.stock_position
        done = terminated or truncated
        total_reward += reward

    return {
        "pnl": env.portfolio_value,
        "total_tc": env.total_tc,
        "total_reward": total_reward,
        "n_trades": n_trades,
    }


def make_rl_policy(model):
    """Adapt a Stable-Baselines3 model to the standard policy signature."""
    def policy(env, obs, prev_h):  # noqa: ARG001
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    """Aggregate episode results into the reported metrics."""
    pnls = np.array([r["pnl"] for r in results])
    tcs = np.array([r["total_tc"] for r in results])
    trades = np.array([r["n_trades"] for r in results])

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
        "avg_trades": float(np.mean(trades)),
    }


# ------------------------------------------------------------------
# Whalley-Wilmott calibration
# ------------------------------------------------------------------

def calibrate_ww(env_kwargs: dict, scenario: str) -> float:
    """Pick the risk aversion maximising held-out Sharpe for this scenario.

    Tuned on ``WW_TUNE_SEEDS``, which is disjoint from the evaluation seeds,
    so the benchmark is given its best honest shot without peeking at the
    episodes it will be scored on.
    """
    best_gamma, best_score = WW_GAMMA_GRID[0], -np.inf
    for g in WW_GAMMA_GRID:
        policy = make_whalley_wilmott(g)
        res = [run_episode(env_kwargs, policy, seed=s) for s in WW_TUNE_SEEDS]
        pnls = np.array([r["pnl"] for r in res])
        sd = pnls.std()
        score = pnls.mean() / sd if sd > 1e-9 else -np.inf
        if score > best_score:
            best_gamma, best_score = g, score
    print(f"  [WW] {scenario}: gamma_risk={best_gamma:g} (held-out Sharpe {best_score:+.4f})")
    return best_gamma


# ------------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------------

def evaluate_all(n_episodes: int = N_EPISODES) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all strategy x scenario combinations with significance testing."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
            print(f"Warning: {path}.zip not found - skipping {name.upper()}")

    rows, pnl_rows, sig_rows = [], [], []

    for scenario_name, env_kwargs in SCENARIOS.items():
        print(f"\n{'='*68}")
        print(f"Scenario: {scenario_name}  ({n_episodes} episodes each)")
        print(f"{'='*68}")

        ww_gamma = calibrate_ww(env_kwargs, scenario_name)

        policies = dict(RULE_BASED)
        policies["whalley_wilmott"] = make_whalley_wilmott(ww_gamma)
        for name in ("ppo", "sac"):
            if models[name] is not None:
                policies[name] = make_rl_policy(models[name])

        pnl_by_strategy: dict[str, np.ndarray] = {}

        for strategy, policy in policies.items():
            np.random.seed(0)  # makes the `random` control reproducible
            episode_results = [
                run_episode(env_kwargs, policy, seed=ep) for ep in range(n_episodes)
            ]

            metrics = compute_metrics(episode_results)
            extra = {"ww_gamma_risk": ww_gamma} if strategy == "whalley_wilmott" else {}
            rows.append({"strategy": strategy, "scenario": scenario_name, **metrics, **extra})

            pnls = np.array([r["pnl"] for r in episode_results])
            pnl_by_strategy[strategy] = pnls
            for ep, r in enumerate(episode_results):
                pnl_rows.append({
                    "scenario": scenario_name, "strategy": strategy, "episode": ep,
                    "seed": ep, "pnl": r["pnl"], "total_tc": r["total_tc"],
                    "n_trades": r["n_trades"],
                })

            print(
                f"  {strategy:16s} | mean={metrics['mean_pnl']:+.4f} "
                f"std={metrics['std_pnl']:.4f} sharpe={metrics['sharpe']:+.3f} "
                f"VaR95={metrics['var_95']:+.4f} CVaR95={metrics['cvar_95']:+.4f} "
                f"tc={metrics['avg_tc']:.4f} trades={metrics['avg_trades']:.1f}"
            )

        print(f"\n  Paired bootstrap vs '{BENCHMARK}' ({N_BOOT:,} resamples):")
        scen_sig = compare_all(pnl_by_strategy, BENCHMARK, n_boot=N_BOOT, seed=0)
        for r in scen_sig:
            r["scenario"] = scenario_name
            sig_rows.append(r)
            if r["metric"] in ("mean_pnl", "sharpe", "cvar_95"):
                star = "*" if r["significant"] else " "
                print(
                    f"    {star} {r['strategy']:16s} {r['metric']:9s} "
                    f"diff={r['diff']:+.4f}  95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
                    f"p={r['p_value']:.4f}"
                )

    df = pd.DataFrame(rows)
    pnl_df = pd.DataFrame(pnl_rows)
    sig_df = pd.DataFrame(sig_rows)[[
        "scenario", "strategy", "benchmark", "metric", "value_a", "value_b",
        "diff", "ci_low", "ci_high", "p_value", "significant",
    ]]

    df.to_csv(os.path.join(RESULTS_DIR, "evaluation_results.csv"), index=False)
    df.to_json(os.path.join(RESULTS_DIR, "evaluation_results.json"), orient="records", indent=2)
    pnl_df.to_csv(os.path.join(RESULTS_DIR, "episode_pnl.csv"), index=False)
    sig_df.to_csv(os.path.join(RESULTS_DIR, "significance.csv"), index=False)

    print(f"\nResults      -> {RESULTS_DIR}/evaluation_results.csv")
    print(f"Per-episode  -> {RESULTS_DIR}/episode_pnl.csv")
    print(f"Significance -> {RESULTS_DIR}/significance.csv")

    return df, pnl_df, sig_df


if __name__ == "__main__":
    df, _, sig = evaluate_all()
    print("\n\nFull Results Table:")
    print(df.to_string(index=False))
