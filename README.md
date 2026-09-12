# RL Derivative Hedging

Reinforcement learning agents (PPO, SAC) trained to dynamically hedge a short European call
position using real SPY market data, benchmarked against Black-Scholes delta hedging and the
Whalley-Wilmott no-transaction band across four market and transaction-cost regimes.

Every comparison is reported with a **paired-bootstrap confidence interval and p-value**, so
null results are legible as null results. The headline finding is not that RL beats
Black-Scholes — against a properly specified cost-aware benchmark it does not — but that the
achievable cost/tail frontier is set by the objective you write down, and that both objectives
tried here are gameable in instructive ways.

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

This is the default (`risk_objective="quadratic"`). It minimises hedging error *in
expectation* and is therefore indifferent to the shape of the loss tail — which is the
measured reason the trained agent shows a wider CVaR than delta hedging (see
[Key Results](#key-results)).

A tail-sensitive alternative (`risk_objective="cvar"`) adds the Rockafellar-Uryasev
representation of Conditional Value at Risk:

```
CVaR_q(V) = min_ν { ν + (1/q) · E[(ν − V)⁺] }
r_T += −λ_c · (1/q) · max(ν − V_T, 0)
```

The inner expectation is an ordinary per-episode quantity, so it drops straight onto the
terminal reward, charging the policy only for outcomes below `ν` and in proportion to how
far below. `ν` is the VaR itself and is unknown in advance, so it is tracked online by the
stochastic quantile update `ν ← ν + lr·(q − 1{V_T < ν})`, whose fixed point is the
q-quantile of the terminal P&L distribution.

---

## Data

**Source:** Yahoo Finance via `yfinance`  
**Coverage:** SPY daily OHLCV, 2021-04-14 → 2026-04-13 (1,255 trading days; 1,235 after the
21-day realised-vol warm-up)  
**Training windows:** 1,205 distinct 30-day overlapping windows  
**Calibrated parameters:** σ = 17.05 %, μ = 11.37 % (annualised over the sample)

The `RealDataHedgingEnv` replays actual historical price windows normalised to S₀ = 100. Each
episode samples a random window, so the agent trains across several distinct regimes — the 2022
rate-shock drawdown, the 2023–24 bull market, and the 2025–26 period are all in the training
distribution. Note that the sample **begins after the March 2020 COVID crash**, so the most
violent recent volatility episode is *not* represented; the `vol_mismatch` and `regime_switch`
evaluation scenarios exist partly to probe that gap synthetically.

Windows overlap and are sampled from the whole history by default (`split="all"`), which
maximises regime coverage but leaks across time. Use `split="train"` / `split="test"` for an
embargoed chronological split when an out-of-sample claim is needed.

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

### Benchmarks

| Strategy | Description |
|---|---|
| `no_hedge` | Hold zero stock — isolates raw short-option exposure |
| `delta` | Rebalance to Black-Scholes delta every step, ignoring cost |
| `whalley_wilmott` | **Cost-aware no-transaction band** (Whalley & Wilmott, 1997) |
| `random` | Uniform random target — control for "did the agent learn anything" |
| `ppo` / `sac` | Trained agents, deterministic rollout |

Naive delta hedging is a weak benchmark: it is cost-*unaware* by construction, so any
cost-sensitive policy beats it once transaction costs bite. The honest benchmark is
**Whalley-Wilmott**, the asymptotically optimal policy under proportional costs for an
agent with exponential utility. It does nothing while the holding stays inside a band
around the BS delta, and trades only to the nearest edge when the band is breached:

```
H = ( 3/2 · e^{-r·τ} · λ · S · Γ² / γ_risk )^{1/3}
no-trade region: [ Δ − H , Δ + H ]
```

The band widens with transaction cost and gamma, and narrows as risk aversion rises;
`H → 0` recovers continuous delta hedging. `γ_risk` is calibrated **per scenario on a
disjoint held-out seed block** (seeds 10 000–10 199) and then frozen for the reported
evaluation seeds (0–449), so the benchmark is given its best honest shot without
peeking at the episodes it is scored on. The grid selected the smallest value tested
(`γ_risk = 0.01`) in every scenario, i.e. the widest bands on the grid — the optimum lies
at or beyond the boundary, so these figures are a *lower* bound on Whalley-Wilmott's
achievable performance, not a tuned optimum.

### Methodology

**Common random numbers.** Episode *i* uses `seed=i` for *every* strategy, so all
strategies are scored on identical price paths. This is a variance-reduction
technique: it removes path noise from strategy differences and makes the per-episode
P&L differences genuinely paired.

**Paired bootstrap.** Because the comparisons are paired, significance is assessed by
resampling *episode indices* (10 000 resamples) and applying the same index set to both
strategies. Resampling each strategy independently would discard the pairing and
inflate the intervals. Every reported difference carries a 95 % percentile CI and a
two-sided bootstrap p-value. VaR and CVaR are nonlinear functionals of the P&L
distribution, which is precisely why the bootstrap is used rather than a closed-form
standard error.

**A note on "Sharpe".** These are hedging portfolios with no alpha source, so mean
terminal P&L is a *cost* and every Sharpe figure is negative. The quantity is
mean P&L / σ(P&L) — risk-normalised hedging cost — and **less negative is better**.
It is not an annualised investment Sharpe ratio.

---

## Results

450 paired episodes per scenario; P&L in dollars per $100 of notional.

### Base — σ_model = σ_realized = 20 %, TC = 0.1 %

| Strategy | Mean P&L | σ | Sharpe | CVaR 95 % | % loss | Avg TC | Trades |
|---|---|---|---|---|---|---|---|
| no_hedge | −0.354 | 4.379 | −0.081 | −12.523 | 37.6 % | 0.000 | 0.0 |
| delta | −0.184 | 1.003 | −0.183 | **−2.449** | 55.3 % | 0.226 | 29.8 |
| whalley_wilmott | **−0.183** | 1.963 | **−0.093** | −4.086 | 48.4 % | **0.070** | **8.9** |
| ppo | **−0.175** | 1.199 | −0.146 | −2.950 | 52.9 % | 0.223 | 30.0 |
| sac | −0.338 | 1.736 | −0.195 | −4.710 | 54.0 % | 0.339 | 29.1 |
| random | −2.538 | 6.089 | −0.417 | −18.171 | 64.2 % | 2.012 | 30.0 |

### High transaction cost — TC = 1 % (10×)

| Strategy | Mean P&L | σ | Sharpe | CVaR 95 % | % loss | Avg TC | Trades |
|---|---|---|---|---|---|---|---|
| delta | −2.220 | 1.108 | −2.004 | **−4.918** | 99.1 % | 2.263 | 29.8 |
| whalley_wilmott | **−0.747** | 3.750 | **−0.199** | −8.020 | **46.4 %** | **0.457** | **3.9** |
| ppo | −2.185 | 1.465 | −1.491 | −5.505 | 94.0 % | 2.233 | 30.0 |
| sac | −3.386 | 2.417 | −1.401 | −8.942 | 95.1 % | 3.387 | 29.1 |

### Volatility mismatch — model 20 %, market realises 30 %

| Strategy | Mean P&L | σ | Sharpe | CVaR 95 % | % loss | Avg TC | Trades |
|---|---|---|---|---|---|---|---|
| delta | −1.638 | 1.642 | −0.998 | **−5.685** | 84.4 % | 0.261 | 29.2 |
| whalley_wilmott | **−1.590** | 2.467 | **−0.644** | −6.677 | 67.8 % | **0.086** | **10.2** |
| ppo | −1.636 | 2.144 | −0.763 | −6.593 | 79.6 % | 0.261 | 30.0 |
| sac | −1.831 | 2.830 | −0.647 | −8.754 | 73.8 % | 0.421 | 28.7 |

### Regime switching — vol alternates 15 % / 35 %

| Strategy | Mean P&L | σ | Sharpe | CVaR 95 % | % loss | Avg TC | Trades |
|---|---|---|---|---|---|---|---|
| delta | −1.038 | 1.724 | −0.602 | **−5.331** | 71.3 % | 0.236 | 29.4 |
| whalley_wilmott | **−0.979** | 2.462 | **−0.398** | −6.344 | 58.2 % | **0.076** | **9.8** |
| ppo | −0.986 | 2.188 | −0.450 | −6.632 | 62.9 % | 0.234 | 30.0 |
| sac | −1.407 | 2.838 | −0.496 | −8.382 | 65.1 % | 0.363 | 28.8 |

---

## Key Results

### 1. PPO improves risk-normalised hedging cost over naive delta — but not in the base case

Sharpe improves significantly in the three stressed regimes (high_tc +0.513, vol_mismatch
+0.235, regime_switch +0.152; all *p* < 0.0001) and **not** in the base case
(+0.037, 95 % CI [−0.020, +0.095], *p* = 0.19). PPO's *mean* P&L advantage over delta is
**not statistically significant in any scenario** (*p* = 0.80 / 0.34 / 0.97 / 0.29) — the
gain is concentrated entirely in variance reduction, not in average cost.

### 2. PPO's tail is significantly worse than delta's — in all four scenarios

95 % CVaR degrades by 0.50 / 0.59 / 0.91 / 1.30 (all *p* ≤ 0.0002). This is not noise and
it is not a bug: the reward is a squared-error objective, which is minimised in
*expectation* and is therefore indifferent to the shape of the loss tail. Two policies
with equal mean squared error score identically even if one occasionally loses five times
as much. The agent optimised exactly what it was told to optimise. See
`utils/risk_objectives.py` for the CVaR-augmented alternative.

### 3. A 1997 closed-form solution matches or beats both RL agents

Whalley-Wilmott improves Sharpe over delta significantly in **all four** scenarios, at
**65–87 % fewer trades** — though the base-case margin is slim (+0.090, *p* = 0.047) and
should be read as borderline rather than established. Head-to-head against PPO it is a statistical tie on mean P&L and
Sharpe in base, vol_mismatch and regime_switch — and in the high-cost regime it is
decisively better (mean +1.44, Sharpe +1.29, both *p* < 0.0001), cutting trading from 30
rebalances to 3.9. Neither RL agent learned the cost-aware waiting behaviour that the band
encodes analytically.

### 4. PPO's one genuine edge over Whalley-Wilmott is tail risk

PPO's CVaR is significantly better than WW's in base (+1.14, *p* < 0.0001) and high_tc
(+2.51, *p* < 0.0001). Because PPO rehedges every step it never lets a gap accumulate,
whereas the band tolerates drift to save cost.

### 5. SAC underperforms

SAC is significantly worse than delta on mean P&L in three of four scenarios and loses to
Whalley-Wilmott on nearly every metric while paying ~50 % more in transaction costs.
Reported here rather than omitted.

### 6. Fixing the objective did not fix the tail — it got gamed

Finding 2 suggests an obvious remedy: replace the squared-error reward with the
tail-sensitive CVaR objective and retrain. That experiment was run
(`--risk-objective cvar --cvar-weight 1.0`, 500k steps, same data and hyperparameters).

**It did not work, and the way it failed is the most interesting result in the project.**

| base scenario | Mean P&L | Sharpe | CVaR 95 % | mean holding *h* |
|---|---|---|---|---|
| delta | −0.184 | −0.183 | **−2.449** | 0.521 |
| ppo (quadratic) | −0.175 | −0.146 | −2.950 | 0.569 |
| ppo (CVaR) | **+0.113** | **+0.056** | −5.298 | 0.789 |

Sharpe improved significantly in all four scenarios (+0.19 to +0.65, all *p* < 0.0001)
and mean P&L turned *positive* in the base case — but 95 % CVaR got **worse in all four**
(−1.11 to −2.54, all *p* ≤ 0.015). The objective designed to shrink the tail roughly
doubled it.

**Diagnosis.** The agent's average deviation from the BS delta is `mean|h − Δ| = 0.2674`
and its *signed* deviation is `mean(h − Δ) = +0.2674` — identical to four decimals. The
deviation is therefore one-directional: the policy never hedges less than delta, it simply
holds a permanent **+0.27 long tilt** on top of the hedge. It is not hedging better; it is
carrying a directional position.

**Causal test.** If that tilt is harvesting the equity risk premium, its advantage must
disappear when the drift does. Re-evaluating the same frozen policy under different μ:

| Drift | CVaR-agent mean P&L advantage vs quadratic agent | *p* | Its CVaR 95 % |
|---|---|---|---|
| μ = +5 % (as trained) | **+0.288** [+0.124, +0.444] | <0.0001 | −5.30 |
| μ = 0 | +0.169 [−0.004, +0.333] | 0.054 | −5.57 |
| μ = −5 % | +0.041 [−0.137, +0.211] | 0.624 | −5.85 |

The edge decays monotonically with drift and vanishes once the drift is removed. The
fattened tail persists regardless.

**Why it happened.** The Rockafellar-Uryasev term penalises terminal outcomes below `ν` —
it is *linear and one-sided*. Under a positive drift, the cheapest way to reduce
`P(V_T < ν)` is not to hedge more precisely but to accumulate long exposure and collect
the risk premium, which lifts the whole P&L distribution above `ν` most of the time while
making the minority of down-paths much worse. The agent optimised the risk measure it was
given rather than the risk it was meant to control.

The lesson generalises past this project: a risk penalty attached to terminal wealth in a
market with drift is an invitation to take directional risk. Making it work requires
removing the drift channel — for example a **drift-neutral (martingale) training measure**,
a penalty on `|h − Δ|` exposure rather than on wealth, or a genuinely distributional critic
that models the whole return distribution instead of a scalar shortfall.

### The honest summary

The policies trace a **transaction-cost / tail-risk frontier and none dominates**: delta
pays full cost for the tightest tail, Whalley-Wilmott pays the least cost and accepts the
widest tail, PPO sits between them. The headline is not "RL beats Black-Scholes" — against
a properly specified classical benchmark it does not — but that **the frontier is set by
the objective you write down**, and that both objectives tried here are gameable in
instructive ways: squared error is blind to the tail, and a naive CVaR penalty is
satisfied by taking directional risk instead of hedging. Every number above is reported
with a paired-bootstrap CI so that the null results are legible as null results.

---

## Reproducing

```bash
python evaluation/full_evaluation.py     # 18k episodes + 10k bootstrap resamples
```

Writes `results/evaluation_results.csv` (aggregates), `results/episode_pnl.csv`
(per-episode P&L, for independent re-analysis) and `results/significance.csv`
(every CI and p-value).

The run is fully deterministic — episode seeds, the bootstrap RNG and the Whalley-Wilmott
calibration grid are all fixed — so a clean re-run reproduces the committed CSVs
byte-for-byte. Takes roughly 15 minutes on a laptop CPU.

> **Caveat on scope.** Training uses `RealDataHedgingEnv` (real SPY windows); the
> scenario evaluation above uses `RLHedgingEnv` (synthetic GBM), because GBM allows σ
> and regime structure to be controlled exactly. The reported figures are therefore a
> **zero-shot transfer** test, not a historical backtest. GBM has no fat tails and no
> volatility clustering, so it likely *understates* the advantage of an adaptive policy.
> `RealDataHedgingEnv(split="train"|"test")` provides an embargoed chronological split
> for a genuine out-of-sample historical evaluation.

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
│   ├── full_evaluation.py     # 6 strategies × 4 scenarios + bootstrap CIs
│   ├── hedging_strategies.py  # delta, Whalley-Wilmott band, controls
│   ├── bootstrap.py           # paired bootstrap significance testing
│   └── gamma_volatility_analysis.py
├── data/
│   └── generate_data.py       # Fetches 5-year SPY history + options chain
├── utils/
│   ├── black_scholes.py
│   ├── market_simulator.py
│   └── risk_objectives.py     # quadratic vs CVaR reward objectives
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
├── results/
│   ├── evaluation_results.csv      # aggregate metrics
│   ├── episode_pnl.csv             # per-episode P&L, for re-analysis
│   ├── significance.csv            # every bootstrap CI and p-value
│   └── cvar_objective_comparison.csv
└── models/
    ├── ppo_hedger.zip              # quadratic objective (baseline)
    ├── ppo_hedger_cvar.zip         # CVaR objective (see Key Results §6)
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

Optional training flags:

```bash
# tail-sensitive objective instead of plain squared error
python training/train_ppo.py --risk-objective cvar --cvar-weight 1.0 \
                             --model-path models/ppo_hedger_cvar

# embargoed chronological split for genuine out-of-sample evaluation
python training/train_ppo.py --split train --train-frac 0.8
```

Models in `models/` are loaded automatically by the app. If a model already exists at the
target path, training continues from that checkpoint at a reduced learning rate rather than
restarting from random weights; delete the `.zip` to train from scratch. Training a variant
via `--model-path` writes its learning curves and checkpoints to a run-specific directory, so
variants never overwrite the baseline run's artefacts.

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
