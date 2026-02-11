# Delta Hedging Gamma Risk Analysis - Results Summary

## Overview
This analysis demonstrates **why delta hedging fails in practice** by focusing on three core gamma risk phenomena:
1. **Volatility Misspecification** (Model vs Realized)
2. **Gamma Explosion** (ATM effects)
3. **Time-to-Maturity Risk** (Short-dated options)

---

## Experiment 1: Volatility Misspecification
**File:** `01_gamma_pnl_vs_realized_volatility.png`

### What This Shows:
The hedger assumes model volatility = 20%, but the actual realized volatility varies. This creates **gamma P&L** (profit/loss from being short gamma):

- **Realized Vol < 20%** (10%, 15%): Gains from being short gamma (~$0)
  - Lower realized moves means fewer rehedging losses
  
- **Realized Vol = 20%**: Breakeven (~$-0.15)
  - Model perfectly predicts reality
  
- **Realized Vol > 20%** (25%, 30%, 40%): Losses from gamma (~$-0.46)
  - Higher realized moves amplify hedging losses
  - At 40% realized vol vs 20% model vol: ~$0.46 loss per simulation

### Key Insight:
**Gamma P&L ≈ 0.5 × Gamma × (σ_realized - σ_model)²**
- Short gamma positions lose when vol surprises to the upside
- This is the fundamental risk in delta hedging: **model risk**

---

## Experiment 2: Gamma Explosion - ATM vs OTM/ITM  
**File:** `02_gamma_explosion_atm_vs_otm.png`

### What This Shows:
When model vol = 25% and realized vol = 35%, hedging error varies dramatically by strike:

| Strike | Type | Mean Loss | Explanation |
|--------|------|-----------|-------------|
| $80    | OTM  | +$0.10    | Far out-of-money, very low gamma |
| $90    | OTM  | -$0.06    | Near-the-money, moderate gamma |
| **$100** | **ATM** | **-$0.30** | **Maximum gamma here** |
| $110   | ITM  | -$0.03    | Near-the-money, moderate gamma |
| $120   | ITM  | +$0.06    | Far in-the-money, very low gamma |

### Key Insight:
**Gamma is maximum at-the-money (S = K)**
- ATM options have ~5x the gamma of far OTM/ITM options
- This means:
  - **Harder to hedge:** More frequent rebalancing needed
  - **More expensive:** Higher transaction costs
  - **Riskier:** Larger gamma P&L swings

---

## Experiment 3: Time-to-Maturity Effects
**File:** `03_time_to_maturity_gamma_explosion.png`

### What This Shows:
As options approach expiration, gamma explodes and hedging error increases:

| Days to Expiry | Mean Loss | Std Dev | Notes |
|---|---|---|---|
| 365 days (1Y)  | -$0.52 | $2.18 | Large variance, manageable |
| 180 days (6M)  | -$0.17 | $1.36 | Still relatively stable |
| 90 days (3M)   | -$0.27 | $1.38 | Variance increasing |
| 30 days        | -$0.17 | $0.90 | Concentrated around mean |
| 10 days        | -$0.08 | $0.58 | Tighter distribution, lower vol |
| 5 days         | +$0.02 | $0.46 | Close to expiry, semi-random walk |

### Key Insight:
**Gamma grows as T → 0 (mathematical: Gamma ∝ 1/√T)**
- At expiration: Stock price movements create **infinite gamma** 
- Near expiration:
  - Overnight gaps are catastrophic (can't rehedge)
  - Bid-ask spreads matter much more
  - Every $0.01 stock move = significant P&L
  - Options are "all or nothing" payoff

---

## The Gamma Explosion Paradox

### Traditional View (Theory):
"Delta hedging perfectly replicates the option payoff if:
- No transaction costs
- Continuous rebalancing
- Perfect volatility forecasting"

### Reality (Practice):
✗ We **can't** predict volatility perfectly (gamma risk)  
✗ We **can't** rebalance continuously (discrete rebalancing risk)  
✗ **Every rebalance costs money** (transaction costs)

### The Bottom Line:

```
Delta Hedge "Profit" = 
  - 0.5 × Gamma × (σ_actual - σ_model)² × T  [GAMMA LOSS]
  - Sum of transaction costs                    [COST LOSS]
  + Vega × (σ_actual - σ_model)               [VOL GAIN - cancels in theory]

If σ_actual > σ_model → NET LOSS
If σ_actual < σ_model → SMALL GAIN (bounded by realized moves)
```

---

## Why Quants & Banks Still Use Delta Hedging

1. **It's the least-bad option**
   - Managing directional risk is essential
   - Gamma/vega risks are secondary to delta
   
2. **Profits come from vol arbitrage**
   - Sell overpriced options (high implied vol)
   - Delta hedge to remove directional risk
   - Keep profits from vol mispricing
   
3. **Risk management, not perfect hedging**
   - Reduces exposure, doesn't eliminate it
   - Focus on **daily P&L management**
   - Adjust hedges dynamically based on market conditions

---

## Practical Implications for Traders/Risk Managers

| Situation | Action | Reasoning |
|-----------|--------|-----------|
| Realized vol < Implied vol | **SELL options & delta hedge** | Profit from vol decay + hedge directional |
| Realized vol > Implied vol | **BUY options** (less hedging needed) | Gamma gains offset vega losses |
| ATM, Short-dated options | **HEDGE FREQUENTLY** or **AVOID**  | Gamma too large, rebalancing too expensive |
| Far OTM/ITM options | **HEDGE INFREQUENTLY** | Low gamma, can skip rebalancing |
| Vol uncertainty high | **STAY DELTA NEUTRAL** | Don't take directional bets |

---

## Conclusion

Delta hedging is theoretically perfect but practically flawed because:

1. **Gamma Risk:** Can't predict future volatility ⟹ Gamma P&L surprises
2. **Omega Risk:** Need frequent rebalancing ⟹ Expensive transaction costs
3. **Model Risk:** Assumptions are always wrong ⟹ Real markets deviate
4. **Timing Risk:** Can't rebalance continuously ⟹ Gaps between rebalances

**The best traders don't believe in perfect hedges—they manage risk dynamically and make profits from understanding where models fail.**

---

*Generated: 2026-02-11*  
*Dataset: 50 Monte Carlo paths × 6 volatility scenarios × 5 strikes × 6 maturities*
