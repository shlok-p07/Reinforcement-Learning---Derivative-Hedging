"""
Gamma & Volatility Analysis Module

Evaluates structural failure of delta hedging under:
- Volatility misspecification
- Gamma explosion (ATM risk)
- Time-to-maturity decay
"""

from typing import Dict, List
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.market_simulator import Simulator
from utils.black_scholes import EuropeanCallOption
from envs.hedging_envs import DeltaHedgingEnv


# ============================================================
# Core Hedging Engine (Model vs Realized Volatility Separation)
# ============================================================

def run_single_hedge_vol_mismatch(
    s0: float,
    mu: float,
    model_sigma: float,
    realized_sigma: float,
    dt: float,
    maturity: float,
    strike: float,
    rate: float,
    transaction_cost: float,
    seed: int | None = None,
) -> float:
    """
    Runs one delta-hedged path where:
    - Market evolves with realized_sigma
    - Hedge uses model_sigma

    Returns final portfolio P&L.
    """

    simulator = Simulator(s0, mu, realized_sigma, dt, seed)

    model_option = EuropeanCallOption(strike, model_sigma, rate)
    real_option = EuropeanCallOption(strike, realized_sigma, rate)

    env = DeltaHedgingEnv(simulator, model_option, maturity, dt, transaction_cost)
    env.reset()

    total_steps = int(maturity / dt)

    for _ in range(total_steps):
        # Market moves
        env.spot = env.simulator.step()
        env.tau -= env.dt

        # Real option repriced with realized volatility
        real_price = real_option.price(env.spot, env.tau)

        # Rehedge using model volatility
        delta_new = model_option.delta(env.spot, env.tau)
        trade = delta_new - env.stock_position

        env.cash -= trade * env.spot
        env.cash -= abs(trade) * env.spot * transaction_cost

        env.stock_position = delta_new
        env.delta = delta_new

        # Portfolio value computed using REAL option price
        env.portfolio_val = (
            env.stock_position * env.spot
            + env.cash
            - real_price
        )

    return env.portfolio_val


# ============================================================
# Monte Carlo Wrapper
# ============================================================

def monte_carlo_vol_mismatch(
    n_paths: int,
    model_sigma: float,
    realized_sigma: float,
    **kwargs,
) -> np.ndarray:
    pnls = []

    for seed in range(n_paths):
        pnl = run_single_hedge_vol_mismatch(
            model_sigma=model_sigma,
            realized_sigma=realized_sigma,
            seed=seed,
            **kwargs,
        )
        pnls.append(pnl)

    return np.array(pnls)


# ============================================================
# Risk Metrics
# ============================================================

def compute_risk_metrics(pnls: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(pnls)),
        "std": float(np.std(pnls)),
        "var_95": float(np.percentile(pnls, 5)),
        "mean_abs": float(np.mean(np.abs(pnls))),
    }


# ============================================================
# Experiments
# ============================================================

def experiment_volatility_misspecification(
    model_sigma: float,
    realized_sigmas: List[float],
    n_paths: int,
    base_kwargs: dict,
) -> Dict[float, np.ndarray]:

    results = {}

    for realized_sigma in realized_sigmas:
        pnls = monte_carlo_vol_mismatch(
            n_paths=n_paths,
            model_sigma=model_sigma,
            realized_sigma=realized_sigma,
            **base_kwargs,
        )
        results[realized_sigma] = pnls

    return results


def experiment_maturity_sensitivity(
    maturities: List[float],
    model_sigma: float,
    realized_sigma: float,
    n_paths: int,
    base_kwargs: dict,
) -> Dict[float, np.ndarray]:

    results = {}

    for maturity in maturities:
        pnls = monte_carlo_vol_mismatch(
            n_paths=n_paths,
            model_sigma=model_sigma,
            realized_sigma=realized_sigma,
            maturity=maturity,
            **base_kwargs,
        )
        results[maturity] = pnls

    return results


# ============================================================
# Plotting
# ============================================================

def plot_gamma_vs_realized_vol(
    results: Dict[float, np.ndarray],
    model_sigma: float,
    output_path: str,
) -> None:

    realized = list(results.keys())
    mean_pnls = [np.mean(results[s]) for s in realized]

    plt.figure(figsize=(8, 5))
    plt.bar(
        [f"{s*100:.0f}%" for s in realized],
        mean_pnls,
        edgecolor="black",
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"Gamma P&L vs Realized Vol (Model Vol = {model_sigma*100:.0f}%)")
    plt.ylabel("Mean P&L")
    plt.xlabel("Realized Volatility")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_maturity_sensitivity(
    results: Dict[float, np.ndarray],
    output_path: str,
) -> None:

    maturities = list(results.keys())
    mean_abs = [np.mean(np.abs(results[m])) for m in maturities]

    plt.figure(figsize=(8, 5))
    plt.plot(maturities, mean_abs, marker="o")
    plt.gca().invert_xaxis()
    plt.title("Mean Absolute Hedging Error vs Maturity")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Mean Absolute P&L")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":

    OUTPUT_DIR = "analysis_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BASE_PARAMS = dict(
        s0=100.0,
        mu=0.05,
        dt=1 / 252,
        maturity=30 / 252,
        strike=100.0,
        rate=0.01,
        transaction_cost=0.001,
    )

    N_PATHS = 100

    # --------------------------
    # Volatility Misspecification
    # --------------------------
    model_sigma = 0.20
    realized_sigmas = [0.10, 0.15, 0.20, 0.25, 0.30]

    vol_results = experiment_volatility_misspecification(
        model_sigma=model_sigma,
        realized_sigmas=realized_sigmas,
        n_paths=N_PATHS,
        base_kwargs=BASE_PARAMS,
    )

    plot_gamma_vs_realized_vol(
        vol_results,
        model_sigma,
        os.path.join(OUTPUT_DIR, "gamma_vs_realized_vol.png"),
    )

    # --------------------------
    # Maturity Sensitivity
    # --------------------------
    maturities = [365/365, 180/365, 90/365, 30/365, 10/365]

    maturity_results = experiment_maturity_sensitivity(
        maturities=maturities,
        model_sigma=0.20,
        realized_sigma=0.30,
        n_paths=N_PATHS,
        base_kwargs=BASE_PARAMS,
    )

    plot_maturity_sensitivity(
        maturity_results,
        os.path.join(OUTPUT_DIR, "gamma_vs_maturity.png"),
    )

    print("Gamma & volatility analysis completed.")