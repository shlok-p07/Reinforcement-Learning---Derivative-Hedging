"""Baseline strategy comparison: no_hedge, delta_hedge, random."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from envs.rl_hedging_env import RLHedgingEnv  # noqa: E402

ENV_KWARGS = dict(
    s0=100.0, mu=0.05, sigma=0.20, dt=1 / 252,
    maturity=30 / 252, strike=100.0, rate=0.01,
    transaction_cost=0.001,
)


def run_episode(strategy: str, seed: int) -> tuple[float, float]:
    """Run one episode. Returns (final_pnl, total_reward)."""
    env = RLHedgingEnv(**ENV_KWARGS, seed=seed)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if strategy == "no_hedge":
            action = np.array([0.0], dtype=np.float32)
        elif strategy == "delta_hedge":
            action = np.array([obs[2]], dtype=np.float32)   # obs[2] = delta
        elif strategy == "random":
            action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    return env.portfolio_value, total_reward


def evaluate_strategy(strat: str, n_episodes: int = 200) -> dict:
    """Monte Carlo evaluation over n_episodes seeds."""
    pnl_list: list[float] = []
    reward_list: list[float] = []
    for seed in range(n_episodes):
        pnl, reward = run_episode(strat, seed)
        pnl_list.append(pnl)
        reward_list.append(reward)

    pnl_arr = np.array(pnl_list)
    return {
        "mean_pnl":    float(np.mean(pnl_arr)),
        "std_pnl":     float(np.std(pnl_arr)),
        "var_95":      float(np.percentile(pnl_arr, 5)),
        "mean_reward": float(np.mean(reward_list)),
    }


if __name__ == "__main__":
    strat_names = ["no_hedge", "delta_hedge", "random"]

    print("\nBaseline Strategy Comparison")
    print("=" * 50)

    for strat_name in strat_names:
        results = evaluate_strategy(strat_name, n_episodes=200)
        print(f"\nStrategy: {strat_name}")
        print(f"  Mean Final P&L : {results['mean_pnl']:+.4f}")
        print(f"  Std Dev P&L    : {results['std_pnl']:.4f}")
        print(f"  95% VaR        : {results['var_95']:+.4f}")
        print(f"  Mean Reward    : {results['mean_reward']:.4f}")
