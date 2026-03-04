import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from envs.rl_hedging_env import RLHedgingEnv

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

# -------- Test 1: No Hedge --------
state = env.reset()
total_reward_no_hedge = 0.0
done = False

while not done:
    action = 0.0
    state, reward, done, _ = env.step(action)
    total_reward_no_hedge += reward

print("Total Reward (No Hedge):", total_reward_no_hedge)


# -------- Test 2: Delta Hedge --------
state = env.reset()
total_reward_delta = 0.0
done = False

while not done:
    action = state[2]  # model delta
    state, reward, done, _ = env.step(action)
    total_reward_delta += reward

print("Total Reward (Delta Hedge):", total_reward_delta)