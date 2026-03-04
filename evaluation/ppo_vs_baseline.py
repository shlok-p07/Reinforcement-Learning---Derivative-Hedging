import os
import sys
import numpy as np
from stable_baselines3 import PPO
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.rl_hedging_env import RLHedgingEnv


MODEL_PATH = "models/ppo_hedger"

N_EPISODES = 200


def evaluate_strategy(strategy_name: str, model=None):
    final_pnls = []

    for episode in range(N_EPISODES):

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

        state, _ = env.reset()
        done = False

        while not done:

            if strategy_name == "ppo":
                action, _ = model.predict(state, deterministic=True)

            elif strategy_name == "delta":
                action = np.array([state[2]])  # delta in state

            elif strategy_name == "no_hedge":
                action = np.array([0.0])

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        final_portfolio_value = (
            env.stock_position * env.spot
            + env.cash
            - env.option.price(env.spot, 1e-8)
        )
        final_pnls.append(final_portfolio_value)

    return np.array(final_pnls)


if __name__ == "__main__":

    print("\nEvaluating PPO vs Baselines")
    print("="*50)

    model = PPO.load(MODEL_PATH)

    strategies = ["no_hedge", "delta", "ppo"]

    for strat in strategies:

        pnls = evaluate_strategy(strat, model)

        print(f"\nStrategy: {strat}")
        print(f"Mean P&L: {np.mean(pnls):.4f}")
        print(f"Std Dev:  {np.std(pnls):.4f}")
        print(f"95% VaR:  {np.percentile(pnls, 5):.4f}")
