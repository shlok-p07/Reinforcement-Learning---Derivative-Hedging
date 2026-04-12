# RL Derivative Hedging

Reinforcement learning agents (PPO, SAC) trained to dynamically hedge a short European call position using real SPY market data. Benchmarked against Black-Scholes delta hedging across multiple market regimes and transaction cost regimes.

---

## Problem Statement

A dealer who sells an at-the-money European call must continuously rebalance a stock position to neutralise delta exposure. Classical delta hedging minimizes instantaneous exposure but ignores transaction costs, discrete rebalancing, and realised-vs-implied vol mismatch. This project frames dynamic hedging as a continuous-action MDP and learns a policy that trades off hedging error against transaction costs directly from historical price data.

---

## Environment

### State Space — 6 normalised features

| Feature | Description |
|---|---|
| `spot / S₀` | Normalised spot price (~1.0 at par) |
| `τ / T` | Fraction of time remaining ∈ [0, 1] |
| `Δ` | Black-Scholes delta N(d₁) ∈ [0, 1] |
| `Γ · S · √τ` | Dimensionless gamma exposure |
| `h_t` | Current stock holding ∈ [−1.5, 1.5] |
| `log(S/K)` | Log-moneyness |

### Action Space

Continuous target hedge ratio `h_t ∈ [−1.5, 1.5]`.

### Reward

```
r_t = −λ_h · (ΔV_t)² − 0.5 λ_h · max(−ΔV_t, 0)²
r_T += −λ_T · V_T²   (terminal settlement penalty)
```

`ΔV_t` is the one-step portfolio P&L. The asymmetric penalty term makes the agent risk-averse on the downside. `λ_T = 5` amplifies the final exposure signal.

---

## Data

**Source:** Yahoo Finance via `yfinance`  
**Coverage:** 5 years of SPY daily OHLCV (~1,254 trading days)  
**Training windows:** 1,204 distinct 30-day overlapping windows  
**Calibrated parameters:** σ = 17.0%, μ = 11.0% (annualised, 5-year period)

The `RealDataHedgingEnv` replays actual historical price windows normalised to S₀ = 100. Each episode the environment samples a random window, so the agent trains across multiple distinct market regimes — the 2020 COVID crash, 2022 rate shock, and the 2023–24 bull market are all in the training distribution.

Options data (live SPY chain, implied vol surface) is fetched separately for the dashboard.

---

## Algorithms

### PPO — Proximal Policy Optimisation
- 8 parallel environments (`DummyVecEnv`)
- `n_steps=2048`, `batch_size=256`, `n_epochs=10`
- `γ=0.99`, `λ_GAE=0.95`, `clip=0.2`, `ε_ent=0.005`
- Network: MLP `[256, 256]`
- 500,000 environment steps per run

### SAC — Soft Actor-Critic
- 2 environments, off-policy replay buffer (200k transitions)
- `batch_size=256`, `τ_polyak=0.005`, automatic entropy tuning
- State-dependent exploration (`use_sde=True`)
- Network: MLP `[256, 256]`
- 300,000 environment steps per run

Both agents support continuation training — subsequent runs fine-tune from the saved checkpoint at a reduced learning rate rather than restarting from random weights.

---

## Evaluation

Performance is measured across four market scenarios: base, high transaction cost, volatility mismatch (σ_model ≠ σ_realised), and regime switching (HMM-style low/high vol). Metrics reported per strategy:

| Metric | Description |
|---|---|
| Sharpe | Mean terminal P&L / std terminal P&L |
| VaR 95% | 5th percentile of terminal P&L distribution |
| CVaR 95% | Expected P&L conditional on breach of VaR |
| Avg TC | Mean cumulative transaction cost per episode |
| % Loss episodes | Fraction of episodes ending with negative P&L |

---

## Project Structure

```
├── envs/
│   ├── real_data_env.py       # Real SPY data environment (primary)
│   └── rl_hedging_env.py      # Synthetic GBM environment (baseline comparison)
├── training/
│   ├── train_ppo.py
│   └── train_sac.py
├── evaluation/
│   └── full_evaluation.py
├── data/
│   └── generate_data.py       # Fetches 5-year SPY history + options chain
├── utils/
│   ├── black_scholes.py
│   └── market_simulator.py
├── app/
│   ├── main.py
│   ├── components/
│   │   ├── charts.py
│   │   └── runner.py
│   └── pages/
│       ├── 1_Live_Demo.py
│       ├── 2_Training.py
│       ├── 3_Evaluation.py
│       ├── 4_Scenario_Lab.py
│       └── 5_Market_Data.py
└── models/
    ├── ppo_hedger.zip
    └── sac_hedger.zip
```

---

## Setup

```bash
pip install -r requirements.txt
python data/generate_data.py        # fetch SPY history and options chain
python training/train_ppo.py        # train PPO (500k steps, ~15 min on CPU)
python training/train_sac.py        # train SAC (300k steps, ~10 min on CPU)
python evaluation/full_evaluation.py
streamlit run app/main.py
```

Models in `models/` are loaded automatically by the app. If they exist, training scripts continue from the current checkpoint.

---

## Dashboard

Five-page Streamlit app:

| Page | Description |
|---|---|
| **Live Demo** | Animate a single episode; compare agent vs delta hedge step-by-step |
| **Training** | Launch training, monitor live reward curve and progress |
| **Evaluation** | Full results table, Sharpe bars, VaR/CVaR, P&L distributions |
| **Scenario Lab** | Monte Carlo comparison across user-defined market parameters |
| **Market Data** | Live SPY price feed, options chain, implied vol surface, regime history |

---

## Key Results

RL agents consistently reduce tail risk (CVaR) relative to delta hedging in high transaction cost and volatility mismatch scenarios — regimes where the BS delta becomes a poor hedge. In the base low-cost scenario, delta hedging remains competitive on Sharpe. The value of RL is most pronounced when the hedger's model assumptions diverge from market reality.
