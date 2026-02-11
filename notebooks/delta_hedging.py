'''
Simulation of delta hedging.
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from utils.market_simulator import Simulator
from utils.black_scholes import EuropeanCallOption
from envs.hedging_envs import DeltaHedgingEnv

S0 = 100.0
MU = 0.05
SIGMA = 0.2
DT = 1 / 252
MATURITY = 1.0
STRIKE = 100.0
RATE = 0.01
TRANSACTION_COST = 0.0
N_STEPS = int(MATURITY / DT)
SIMULATOR = Simulator(s0=S0, mu=MU, sigma=SIGMA, dt=DT, seed=42)
OPTION = EuropeanCallOption(STRIKE, SIGMA,RATE)
ENV = DeltaHedgingEnv(
    SIMULATOR,
    OPTION,
    MATURITY,
    DT,
    TRANSACTION_COST,
)
STATES = []
STATE = ENV.reset()
STATES.append(STATE)

for _ in range(N_STEPS):
    STATE = ENV.step()
    STATES.append(STATE)
# --- Extract Time Series ---
SPOTS = [state["spot"] for state in STATES]
PORTFOLIO_VALUES = [state["portfolio_val"] for state in STATES]
STOCK_POSITIONS = [state["stock_position"] for state in STATES]
OPTION_PRICES = [state["option_price"] for state in STATES]
TIMES = np.linspace(0, MATURITY, len(STATES))
# --- Plot Results ---
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(TIMES, SPOTS)
plt.title("Spot Price Over Time")
plt.xlabel("Time (Years)")
plt.ylabel("Spot Price")

plt.subplot(2, 2, 2)
plt.plot(TIMES, OPTION_PRICES)
plt.title("Option Price Over Time")
plt.xlabel("Time (Years)")
plt.ylabel("Option Price")

plt.subplot(2, 2, 3)
plt.plot(TIMES, STOCK_POSITIONS)
plt.title("Hedge Position (Delta) Over Time")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Position")

plt.subplot(2, 2, 4)
plt.plot(TIMES, PORTFOLIO_VALUES)
plt.title("Delta-Hedged Portfolio Value")
plt.xlabel("Time (Years)")
plt.ylabel("Portfolio Value")

plt.tight_layout()
plt.show()

# --- Hedging Error Metrics ---
PORTFOLIO_VALUE_ARRAY = np.array(PORTFOLIO_VALUES)
INITIAL_VALUE = PORTFOLIO_VALUE_ARRAY[0]
PNL = PORTFOLIO_VALUE_ARRAY - INITIAL_VALUE

print("Final P&L:", PNL[-1])
print("P&L Standard Deviation:", np.std(PNL))
print("Maximum Drawdown:", np.min(PNL))
