import os
import sys

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from envs.rl_hedging_env import RLHedgingEnv


def make_env():
    return RLHedgingEnv(
        s0=100.0,
        mu=0.05,
        sigma=0.2,
        dt=1/252,
        maturity=30/252,
        strike=100.0,
        rate=0.01,
        transaction_cost=0.001,
    )


if __name__ == "__main__":

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    model.learn(total_timesteps=100_000)

    model.save("models/ppo_hedger")

    print("Training complete.")
