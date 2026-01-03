'''
Testing market simulator operations.
'''
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from utils.market_simulator import Simulator

s = Simulator(s0=100,mu=0.05,sigma=0.2,
              dt=1/300,seed=42)
prices = [s.reset()]
for _ in range(300):
    prices.append(s.step())
plt.plot(prices)
plt.title("Simulated Asset Price based on Geometric Brownian Motion")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.show()
