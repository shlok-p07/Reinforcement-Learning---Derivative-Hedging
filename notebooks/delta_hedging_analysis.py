"""
Day 3.2 — Why Delta Hedging Fails
Enhanced Monte Carlo Analysis with Gamma Explosion & Volatility Misspecification

This script analyzes:
1. Volatility misspecification: Model volatility vs realized volatility
2. Gamma explosion: Hedging error near ATM with high vol and short maturity
3. Time-to-maturity effects: Hedging becomes harder as option approaches expiration
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

from utils.market_simulator import Simulator
from utils.black_scholes import EuropeanCallOption
from envs.hedging_env import DeltaHedgingEnv

print("✓ Imports successful")

# Base Parameters
S0 = 100.0
MU = 0.05
SIGMA = 0.2
DT = 1 / 252
MATURITY = 30 / 252  # 30 days instead of 1 year for faster execution
STRIKE = 100.0
RATE = 0.01
TRANSACTION_COST = 0.001
N_PATHS = 50  # Reduced for faster execution

print(f"Stock Price: ${S0}")
print(f"Model Volatility: {SIGMA*100}%")
print(f"Maturity: {MATURITY} year")
print(f"Strike: ${STRIKE}")
print(f"Time Step: {DT:.6f} (daily)")
print(f"# of Simulation Paths: {N_PATHS}")

# Core simulation function with separate model vs realized volatility
def run_single_hedge_with_vol_separation(
    s0: float,
    mu: float,
    model_sigma: float,
    realized_sigma: float,  # Actual volatility experienced
    dt: float,
    maturity: float,
    strike: float,
    rate: float,
    transaction_cost: float,
    seed: int | None = None,
    rebalancing_frequency: int = 1,
) -> float:
    """
    Runs delta hedge with separate model and realized volatility.
    
    The hedge is calculated using model_sigma (assumed volatility),
    but the market evolves according to realized_sigma (actual volatility).
    This creates gamma risk.
    """
    # Market evolves with realized volatility
    simulator = Simulator(s0, mu, realized_sigma, dt, seed)
    
    # Hedger uses model volatility to compute delta
    model_option = EuropeanCallOption(strike, model_sigma, rate)
    real_option = EuropeanCallOption(strike, realized_sigma, rate)
    
    # Environment tracks true prices but hedges based on model
    class MismatchedHedgingEnv(DeltaHedgingEnv):
        def __init__(self, sim, model_opt, real_opt, maturity, dt, trans_cost):
            self.simulator = sim
            self.model_option = model_opt
            self.real_option = real_opt
            self.maturity = maturity
            self.dt = dt
            self.trans_cost = trans_cost
            self.portfolio_val = 0.0
            self.reset()
        
        def reset(self):
            self.tau = self.maturity
            self.spot = self.simulator.reset()
            self.real_option_price = self.real_option.price(self.spot, self.tau)
            # Delta based on model (incorrect if model_sigma != realized_sigma)
            self.delta = self.model_option.delta(self.spot, self.tau)
            self.stock_position = self.delta
            self.cash = self.real_option_price - self.stock_position * self.spot
            return self.get_state()
        
        def step(self):
            self.spot = self.simulator.step()
            self.tau -= self.dt
            # Real option price moves with realized volatility
            self.real_option_price = self.real_option.price(self.spot, self.tau)
            # But we rehedge based on model volatility (mismatch!)
            delta_new = self.model_option.delta(self.spot, self.tau)
            trade = delta_new - self.stock_position
            trading_val = abs(trade) * self.spot
            self.cash -= trade * self.spot
            self.cash -= trading_val * self.trans_cost
            self.stock_position = delta_new
            self.delta = delta_new
            self.portfolio_val = self.computeportfolio_val()
            return self.get_state()
        
        def computeportfolio_val(self):
            return self.stock_position * self.spot + self.cash - self.real_option_price
        
        def get_state(self):
            return {
                "spot": self.spot,
                "tau": self.tau,
                "option_price": self.real_option_price,
                "delta": self.delta,
                "stock_position": self.stock_position,
                "cash": self.cash,
                "portfolio_val": self.portfolio_val,
            }
    
    env = MismatchedHedgingEnv(simulator, model_option, real_option, maturity, dt, transaction_cost)
    env.reset()
    
    total_steps = int(maturity / dt)
    for step_num in range(total_steps):
        if step_num % rebalancing_frequency == 0:
            env.step()
        else:
            env.spot = env.simulator.step()
            env.tau -= env.dt
            env.real_option_price = env.real_option.price(env.spot, env.tau)
    
    env.portfolio_val = env.computeportfolio_val()
    return env.portfolio_val


# Monte Carlo with volatility separation
def run_monte_carlo_vol_mismatch(n_paths: int, model_sigma: float, realized_sigma: float) -> np.ndarray:
    """Runs multiple hedging paths with volatility mismatch."""
    pnls = []
    for path_id in range(n_paths):
        pnl = run_single_hedge_with_vol_separation(
            s0=S0,
            mu=MU,
            model_sigma=model_sigma,
            realized_sigma=realized_sigma,
            dt=DT,
            maturity=MATURITY,
            strike=STRIKE,
            rate=RATE,
            transaction_cost=TRANSACTION_COST,
            seed=path_id,
            rebalancing_frequency=1,
        )
        pnls.append(pnl)
    return np.array(pnls)


print("✓ Core functions defined")

# Set matplotlib to save instead of show
plt.ioff()  # Turn off interactive mode

output_dir = "/Users/main/RL-Derivative Hedging/Reinforcement-Learning---Derivative-Hedging/analysis_results"
os.makedirs(output_dir, exist_ok=True)

def save_and_show(filename):
    """Save figure and close."""
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {filename}")
    plt.close('all')

# ============================================================================
# EXPERIMENT 1: Volatility Misspecification
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 1: Volatility Misspecification")
print("="*70)
print("Model assumes 20% volatility, but actual volatility varies.")
print("Higher realized vol → Gamma losses. Lower realized vol → Gamma gains.")
print()

model_sigma = 0.20
realized_sigmas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
vol_mismatch_results = {}

for realized_sig in realized_sigmas:
    print(f"Model Vol: {model_sigma*100:.0f}% | Realized Vol: {realized_sig*100:.0f}%...", end=" ")
    pnls = run_monte_carlo_vol_mismatch(N_PATHS, model_sigma, realized_sig)
    vol_mismatch_results[realized_sig] = pnls
    gamma_loss = np.mean(pnls)
    print(f"Gamma P&L: ${gamma_loss:>8.2f} (±${np.std(pnls):>6.2f})")

print("\n" + "-"*70)
print("Key Insight: Gamma P&L vs Realized Volatility")
print("-"*70)
realized_vols = list(vol_mismatch_results.keys())
gamma_pnls = [np.mean(pnls) for pnls in vol_mismatch_results.values()]
vol_diffs = [100 * (rv - model_sigma) for rv in realized_vols]

plt.figure(figsize=(10, 6))
colors = ['red' if gp < 0 else 'green' for gp in gamma_pnls]
bars = plt.bar(range(len(realized_vols)), gamma_pnls, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
plt.xticks(range(len(realized_vols)), [f'{v*100:.0f}%' for v in realized_vols])
plt.xlabel('Realized Volatility', fontsize=12, fontweight='bold')
plt.ylabel('Gamma P&L ($)', fontsize=12, fontweight='bold')
plt.title(f'Gamma Explosion: Model Vol = {model_sigma*100:.0f}%\n(Negative = Hedging Loss from Gamma Risk)', 
          fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for bar, pnl in zip(bars, gamma_pnls):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'${pnl:.0f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
plt.tight_layout()
save_and_show("01_gamma_pnl_vs_realized_volatility.png")

# ============================================================================
# EXPERIMENT 2: Gamma Explosion - Stock Price Sensitivity
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 2: Gamma Explosion - ATM vs OTM/ITM")
print("="*70)
print("Hedging error increases dramatically near strike (highest gamma).")
print()

model_sigma = 0.25
realized_sigma = 0.35  # Higher realized vol magnifies gamma losses
strikes_to_test = [80, 90, 100, 110, 120]  # 80=far OTM, 120=far ITM (relative to S0=100)
gamma_error_results = {}

for strike in strikes_to_test:
    pnls = []
    for path_id in range(N_PATHS):
        pnl = run_single_hedge_with_vol_separation(
            s0=S0,
            mu=MU,
            model_sigma=model_sigma,
            realized_sigma=realized_sigma,
            dt=DT,
            maturity=MATURITY,
            strike=strike,
            rate=RATE,
            transaction_cost=TRANSACTION_COST,
            seed=path_id,
            rebalancing_frequency=1,
        )
        pnls.append(pnl)
    gamma_error_results[strike] = np.array(pnls)
    moneyness = 100 * (S0 / strike)
    print(f"Strike ${strike} (Moneyness: {moneyness:.0f}%) - Mean Loss: ${np.mean(pnls):.2f}")

# Plot gamma explosion by moneyness
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Distribution plot
strikes = sorted(gamma_error_results.keys())
for strike in strikes:
    pnls = gamma_error_results[strike]
    ax1.hist(pnls, bins=25, alpha=0.5, label=f'Strike ${strike}', edgecolor='black')

ax1.set_xlabel('P&L ($)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Gamma Risk Distributions at Different Strikes\n(Model Vol=25%, Realized Vol=35%)',
              fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Box plot showing error magnitudes
strike_labels = [f'${s}' for s in strikes]
mean_losses = [np.abs(np.mean(gamma_error_results[s])) for s in strikes]
std_losses = [np.std(gamma_error_results[s]) for s in strikes]

ax2.errorbar(range(len(strikes)), mean_losses, yerr=std_losses, 
             marker='o', linestyle='-', linewidth=2, markersize=10, 
             capsize=10, capthick=2, color='steelblue', ecolor='darkblue')
ax2.set_xticks(range(len(strikes)))
ax2.set_xticklabels(strike_labels)
ax2.set_xlabel('Strike Price', fontsize=11, fontweight='bold')
ax2.set_ylabel('Mean Absolute Hedging Loss ($)', fontsize=11, fontweight='bold')
ax2.set_title('Gamma Loss vs Moneyness\n(Peak at ATM = Strike = Spot)',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_and_show("02_gamma_explosion_atm_vs_otm.png")

# ============================================================================
# EXPERIMENT 3: Time-to-Maturity Effect (Short-Dated Options Blow Up)
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 3: Hedging Error vs Time-to-Maturity")
print("="*70)
print("Gamma explosion accelerates as option approaches expiration.")
print("Short-dated options have much higher hedging risk.")
print()

maturities_in_days = [365, 180, 90, 30, 10, 5]  # 1 year down to 5 days
maturity_error_results = {}

for maturity_days in maturities_in_days:
    maturity_years = maturity_days / 365
    pnls = []
    for path_id in range(N_PATHS):
        pnl = run_single_hedge_with_vol_separation(
            s0=S0,
            mu=MU,
            model_sigma=0.20,
            realized_sigma=0.30,  # Volatility surprise
            dt=DT,
            maturity=maturity_years,
            strike=STRIKE,
            rate=RATE,
            transaction_cost=TRANSACTION_COST,
            seed=path_id,
            rebalancing_frequency=1,
        )
        pnls.append(pnl)
    maturity_error_results[maturity_days] = np.array(pnls)
    print(f"Maturity {maturity_days:>3d} days - Mean Loss: ${np.mean(pnls):>8.2f}, Std: ${np.std(pnls):>7.2f}")

# Plot time-to-maturity decay of hedging effectiveness
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Mean absolute error vs maturity
maturities_sorted = sorted(maturity_error_results.keys())
mean_abs_errors = [np.abs(np.mean(maturity_error_results[m])) for m in maturities_sorted]
std_errors = [np.std(maturity_error_results[m]) for m in maturities_sorted]

ax = axes[0, 0]
ax.errorbar(maturities_sorted, mean_abs_errors, yerr=std_errors, 
            marker='o', linestyle='-', linewidth=2.5, markersize=10,
            capsize=10, capthick=2, color='crimson', ecolor='darkred')
ax.set_xlabel('Days to Maturity', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Absolute Hedging Loss ($)', fontsize=11, fontweight='bold')
ax.set_title('Gamma Explosion Near Expiration', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.invert_xaxis()  # Short maturity on the right

# Subplot 2: P&L distributions for selected maturities
ax = axes[0, 1]
for mat_days in [365, 90, 10]:
    if mat_days in maturities_sorted:
        pnls = maturity_error_results[mat_days]
        ax.hist(pnls, bins=20, alpha=0.6, label=f'{mat_days} days', edgecolor='black')
ax.set_xlabel('P&L ($)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('P&L Distributions: 1Y vs 3M vs 10 Days', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 3: Coefficient of Variation (Relative Error)
ax = axes[1, 0]
cv_errors = [np.std(maturity_error_results[m]) / (np.abs(np.mean(maturity_error_results[m])) + 1e-6) 
             for m in maturities_sorted]
ax.plot(maturities_sorted, cv_errors, marker='s', linestyle='-', linewidth=2.5,
        markersize=10, color='darkgreen')
ax.set_xlabel('Days to Maturity', fontsize=11, fontweight='bold')
ax.set_ylabel('Relative Volatility (Std/Mean)', fontsize=11, fontweight='bold')
ax.set_title('Hedging Uncertainty Grows as Expiration Approaches', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

# Subplot 4: Risk metrics table
ax = axes[1, 1]
ax.axis('off')

table_data = []
table_data.append(['Days to Expiry', 'Mean P&L', 'Std Dev', '95% VaR'])
for mat_days in maturities_sorted:
    pnls = maturity_error_results[mat_days]
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    var_95 = np.percentile(pnls, 5)  # 5th percentile
    table_data.append([
        f'{mat_days}',
        f'${mean_pnl:.0f}',
        f'${std_pnl:.0f}',
        f'${var_95:.0f}'
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Header styling
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    color = '#f0f0f0' if i % 2 == 0 else 'white'
    for j in range(4):
        table[(i, j)].set_facecolor(color)

ax.set_title('Risk Metrics Summary', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
save_and_show("03_time_to_maturity_gamma_explosion.png")

# ============================================================================
# Summary Conclusions
# ============================================================================
print("\n" + "="*70)
print("KEY FINDINGS: Why Delta Hedging Fails in Practice")
print("="*70)
print("""
1. GAMMA RISK (Volatility Misspecification):
   • If realized vol > model vol → LARGE LOSSES (concave payoff)
   • If realized vol < model vol → Small gains (convex cost)
   • Gamma P&L ≈ 0.5 × Gamma × (dS)² = (0.5 × Gamma × σ_realized²) × dt
   • Hedger is SHORT gamma when vol is underestimated

2. GAMMA EXPLOSION NEAR STRIKES:
   • Maximum gamma occurs when S ≈ K (at-the-money)
   • Far OTM/ITM options have lower gamma, safer to hedge
   • High implied volatility compounds gamma risk
   • Gamma grows as time to maturity decreases (vega~0 near expiry)

3. SHORT-DATED OPTIONS ARE DANGEROUS:
   • Gamma → ∞ as T → 0 (1/sqrt(T) behavior)
   • 5-day option has ~3x the gamma of 30-day option
   • Tiny stock price moves create huge P&L swings
   • Discrete rebalancing becomes insufficient

4. THE HEDGING PARADOX:
   • Delta neutrality in theory, but gamma risk in practice
   • You're hedging directional risk but taking on volatility risk
   • Transaction costs make frequent rebalancing expensive
   • Infrequent rebalancing leaves gamma exposure
   
   → There is NO PERFECT HEDGE in real markets
""")
print("="*70)
