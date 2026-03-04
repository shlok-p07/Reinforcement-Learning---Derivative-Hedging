import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from envs.rl_hedging_env import RLHedgingEnv


# ============================================================
# Strategy Functions
# ============================================================

def run_episode(env: RLHedgingEnv, strategy: str):
    state = env.reset()
    done = False

    total_reward = 0.0

    while not done:

        if strategy == "no_hedge":
            action = 0.0

        elif strategy == "delta_hedge":
            action = state[2]  # model delta

        elif strategy == "random":
            action = np.random.uniform(-1.0, 1.0)

        else:
            raise ValueError("Unknown strategy")

        state, reward, done, _ = env.step(action)
        total_reward += reward

    final_portfolio_value = (
        env.stock_position * env.spot
        + env.cash
        - env.option.price(env.spot, 1e-8)
    )

    return final_portfolio_value, total_reward


# ============================================================
# Monte Carlo Benchmark
# ============================================================

def evaluate_strategy(strategy: str, n_episodes: int = 200):

    final_pnls = []
    total_rewards = []

    for seed in range(n_episodes):

        env = RLHedgingEnv(
            s0=100.0,
            mu=0.05,
            sigma=0.2,
            dt=1/252,
            maturity=30/252,
            strike=100.0,
            rate=0.01,
            transaction_cost=0.001,
        )

        pnl, reward = run_episode(env, strategy)

        final_pnls.append(pnl)
        total_rewards.append(reward)

    final_pnls = np.array(final_pnls)
    total_rewards = np.array(total_rewards)

    return {
        "mean_pnl": np.mean(final_pnls),
        "std_pnl": np.std(final_pnls),
        "var_95": np.percentile(final_pnls, 5),
        "mean_total_reward": np.mean(total_rewards),
    }


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    strategies = ["no_hedge", "delta_hedge", "random"]

    print("\nBaseline Strategy Comparison")
    print("=" * 50)

    for strategy in strategies:

        results = evaluate_strategy(strategy, n_episodes=200)

        print(f"\nStrategy: {strategy}")
        print(f"Mean Final P&L:       {results['mean_pnl']:.4f}")
        print(f"Std Dev P&L:          {results['std_pnl']:.4f}")
        print(f"95% VaR:              {results['var_95']:.4f}")
        print(f"Mean Total Reward:    {results['mean_total_reward']:.4f}")
