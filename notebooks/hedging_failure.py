"""
Why Delta Hedging Fails?
Monte Carlo analysis of hedging error under realistic constraints.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

from utils.market_simulator import Simulator
from utils.black_scholes import EuropeanCallOption
from envs.hedging_envs import DeltaHedgingEnv


# 2. Base Parameters
S0 = 100.0
MU = 0.05
SIGMA = 0.2
DT = 1 / 252
MATURITY = 1.0
STRIKE = 100.0
RATE = 0.01
TRANSACTION_COST = 0.001
N_PATHS = 500


# 3. Single-Path Delta Hedge Function
def run_single_hedge(
    s0: float,
    mu: float,
    sigma: float,
    dt: float,
    maturity: float,
    strike: float,
    rate: float,
    transaction_cost: float,
    seed: int | None = None,
    rebalancing_frequency: int = 1,
) -> float:
    """
    Runs one delta-hedged path and returns final P&L.

    Steps:
    1. Initialize simulator and option
    2. Create DeltaHedgingEnv
    3. Step through time until maturity
    4. Return final portfolio value
    """
    # Initialize simulator and option
    simulator = Simulator(s0, mu, sigma, dt, seed)
    option = EuropeanCallOption(strike, sigma, rate)
    
    # Create delta hedging environment
    env = DeltaHedgingEnv(simulator, option, maturity, dt, transaction_cost)
    env.reset()
    
    # Determine total number of steps
    total_steps = int(maturity / dt)
    
    # Step through time with specified rebalancing frequency
    for step_num in range(total_steps):
        if step_num % rebalancing_frequency == 0:
            # Rebalance: let the env handle delta calculation and trading
            env.step()
        else:
            # Don't rebalance: just let the market move forward
            env.spot = env.simulator.step()
            env.tau -= env.dt
            env.option_price = env.option.price(env.spot, env.tau)
            # stock_position and cash remain unchanged (no rebalancing, no trading cost)  
    # Return final portfolio value
    env.portfolio_val = env.computeportfolio_val()
    return env.portfolio_val

# 4. Monte Carlo Simulation
def run_monte_carlo(n_paths: int, **kwargs) -> np.ndarray:
    """
    Runs multiple hedging paths and returns array of final P&Ls.
    """
    pnls = []
    for path_id in range(n_paths):
        pnl = run_single_hedge(
            s0=S0,
            mu=MU,
            sigma=SIGMA,
            dt=DT,
            maturity=MATURITY,
            strike=STRIKE,
            rate=RATE,
            transaction_cost=TRANSACTION_COST,
            seed=path_id,  # Different seed for each path
            **kwargs
        )
        pnls.append(pnl)
    
    return np.array(pnls)


# 5. Experiment 1 — Rebalancing Frequency
def experiment_rebalancing_frequencies(frequencies: list[int]) -> dict[int, np.ndarray]:
    """
    Tests hedging error across different rebalancing frequencies.
    
    frequencies: list of integers representing steps between rebalances
                 e.g., [1, 5, 10] means daily, weekly, bi-weekly
    """
    results = {}
    for freq in frequencies:
        pnls = run_monte_carlo(N_PATHS, rebalancing_frequency=freq)
        results[freq] = pnls
    
    return results


# 6. Experiment 2 — Transaction Costs
def experiment_transaction_costs(costs: list[float]) -> dict[float, np.ndarray]:
    """
    Tests hedging error across different transaction cost levels.
    """
    results = {}
    for cost in costs:
        pnls = run_monte_carlo(
            N_PATHS,
            transaction_cost=cost
        )
        results[cost] = pnls
    
    return results


# 7. Experiment 3 — Volatility Stress
def experiment_volatility_stress(sigmas: list[float]) -> dict[float, np.ndarray]:
    """
    Tests hedging error across different volatility levels.
    """
    results = {}
    for sigma in sigmas:
        pnls = run_monte_carlo(
            N_PATHS,
            sigma=sigma
        )
        results[sigma] = pnls
    
    return results


# 8. Results & Plots
def plot_pnl_distributions(results: dict, title: str) -> None:
    """
    Plots histogram of P&L distributions.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    
    # Handle single subplot case
    if len(results) == 1:
        axes = [axes]
    
    for ax, (param, pnls) in zip(axes, results.items()):
        ax.hist(pnls, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('P&L')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{param}')
        ax.axvline(np.mean(pnls), color='red', linestyle='--', label=f'Mean: {np.mean(pnls):.2f}')
        ax.legend()
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_error_vs_parameter(x_values: list, mean_errors: list, xlabel: str, title: str) -> None:
    """
    Plots mean hedging error vs experiment parameter.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, mean_errors, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Mean Absolute P&L', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()