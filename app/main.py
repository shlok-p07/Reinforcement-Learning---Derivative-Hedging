"""RL Derivative Hedging — Research Platform Homepage"""

import os
import sys
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
from components.charts import inject_css, STRATEGY_COLOR  # noqa: E402

st.set_page_config(
    page_title="RL Hedging | Research Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(inject_css(), unsafe_allow_html=True)

ROOT        = os.path.join(os.path.dirname(__file__), "..")
RESULTS_CSV = os.path.join(ROOT, "results", "evaluation_results.csv")
MODEL_DIR   = os.path.join(ROOT, "models")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _model_status(name: str) -> tuple[bool, str]:
    exists = os.path.exists(os.path.join(MODEL_DIR, f"{name}_hedger.zip"))
    badge  = '<span class="badge-live">● LIVE</span>' if exists else '<span class="badge-off">○ NOT TRAINED</span>'
    return exists, badge


def _load_results() -> pd.DataFrame | None:
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV)
    return None


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## RL HEDGING PLATFORM")
    st.markdown('<hr class="ws-divider">', unsafe_allow_html=True)

    ppo_ok, ppo_badge = _model_status("ppo")
    sac_ok, sac_badge = _model_status("sac")
    results_ok = os.path.exists(RESULTS_CSV)

    st.markdown("**System Status**")
    st.markdown(f"PPO Agent &nbsp;&nbsp; {ppo_badge}", unsafe_allow_html=True)
    st.markdown(f"SAC Agent &nbsp;&nbsp; {sac_badge}", unsafe_allow_html=True)
    eval_badge = '<span class="badge-live">● READY</span>' if results_ok else '<span class="badge-warn">◐ PENDING</span>'
    st.markdown(f"Eval Data &nbsp;&nbsp;&nbsp; {eval_badge}", unsafe_allow_html=True)

    st.markdown('<hr class="ws-divider">', unsafe_allow_html=True)

    st.markdown("**Data Coverage**")
    data_dir = os.path.join(ROOT, "data")
    spy_csv  = os.path.join(data_dir, "spy_daily.csv")
    if os.path.exists(spy_csv):
        try:
            spy_df = pd.read_csv(spy_csv)
            st.markdown(
                f'<div style="font-size:11px;color:#6b7a8d;line-height:1.8;">'
                f'Ticker: <b style="color:#d4d8e2;">SPY</b><br>'
                f'Trading days: <b style="color:#d4d8e2;">{len(spy_df):,}</b><br>'
                f'From: <b style="color:#d4d8e2;">{spy_df["date"].iloc[0]}</b><br>'
                f'To: <b style="color:#d4d8e2;">{spy_df["date"].iloc[-1]}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.caption("SPY data loaded.")
    else:
        st.caption("No data. Run `python data/generate_data.py`")

    st.markdown('<hr class="ws-divider">', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:10px;color:#4a5568;letter-spacing:1px;">'
        'RL DERIVATIVE HEDGING · RESEARCH PLATFORM</span>',
        unsafe_allow_html=True,
    )


# ── Page Header ──────────────────────────────────────────────────────────────

col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown('<div class="ws-title">RL DERIVATIVE HEDGING RESEARCH PLATFORM</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ws-subtitle">'
        'Training reinforcement learning agents to outperform classical Black-Scholes delta hedging '
        '— on real SPY market data, across transaction cost regimes, volatility mismatches, and market crashes.'
        '</div>',
        unsafe_allow_html=True,
    )
with col_time:
    now = datetime.datetime.now()
    st.markdown(
        f'<div style="text-align:right;padding-top:10px;">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#6b7a8d;">'
        f'{now.strftime("%Y-%m-%d %H:%M")}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="ws-divider">', unsafe_allow_html=True)


# ── What This Platform Does ───────────────────────────────────────────────────

st.markdown('<div class="ws-section-header">WHAT THIS PLATFORM DOES</div>', unsafe_allow_html=True)

prob_cols = st.columns(3)

with prob_cols[0]:
    st.markdown("""
    <div class="ws-card" style="min-height:200px;">
      <div class="ws-card-title" style="color:#ff4b4b;">THE PROBLEM</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.8;margin-top:8px;">
        A dealer who sells a European call option must continuously rebalance a stock
        position to stay delta-neutral. Classical <b>Black-Scholes delta hedging</b>
        minimizes instantaneous Greek exposure but ignores:<br><br>
        <span style="color:#ff4b4b;">✗</span> Transaction costs eroding P&amp;L<br>
        <span style="color:#ff4b4b;">✗</span> Discrete rebalancing errors<br>
        <span style="color:#ff4b4b;">✗</span> Implied vs realized vol mismatch<br>
        <span style="color:#ff4b4b;">✗</span> Regime changes (crash, recovery)
      </div>
    </div>
    """, unsafe_allow_html=True)

with prob_cols[1]:
    st.markdown("""
    <div class="ws-card" style="min-height:200px;">
      <div class="ws-card-title" style="color:#00d4aa;">OUR APPROACH</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.8;margin-top:8px;">
        We frame dynamic hedging as a <b>continuous-action Markov Decision Process</b>
        and train two RL agents — PPO and SAC — directly on 5 years of real
        SPY price history (1,200+ distinct 30-day windows).<br><br>
        <span style="color:#00d4aa;">✓</span> Learns cost-aware rebalancing<br>
        <span style="color:#00d4aa;">✓</span> Adapts to volatile regimes<br>
        <span style="color:#00d4aa;">✓</span> No model assumptions required<br>
        <span style="color:#00d4aa;">✓</span> Reduces tail risk (CVaR)
      </div>
    </div>
    """, unsafe_allow_html=True)

with prob_cols[2]:
    st.markdown("""
    <div class="ws-card" style="min-height:200px;">
      <div class="ws-card-title" style="color:#4a9eff;">KEY FINDING</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.8;margin-top:8px;">
        RL agents outperform delta hedging most significantly when the
        <b>hedger's model assumptions diverge from market reality</b> — exactly
        the scenarios that matter most in practice.<br><br>
        <span style="color:#4a9eff;">→</span> High TC: RL rebalances less, saves cost<br>
        <span style="color:#4a9eff;">→</span> Vol mismatch: RL adapts dynamically<br>
        <span style="color:#4a9eff;">→</span> Regime switch: RL reduces tail loss<br>
        <span style="color:#4a9eff;">→</span> Base: delta hedging remains competitive
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Live KPI Strip ────────────────────────────────────────────────────────────

df = _load_results()

if df is not None:
    st.markdown('<div class="ws-section-header">LIVE PERFORMANCE — BASE SCENARIO (200 EPISODES)</div>',
                unsafe_allow_html=True)
    base = df[df["scenario"] == "base"]
    strategies_show = [s for s in ["no_hedge", "delta", "ppo", "sac"] if s in base["strategy"].values]
    cols = st.columns(len(strategies_show))

    for col, strat in zip(cols, strategies_show):
        _rows = base[base["strategy"] == strat]
        if _rows.empty:
            continue
        row   = _rows.iloc[0]
        color = STRATEGY_COLOR.get(strat, "#d4d8e2")
        pnl_cls    = "positive" if row["mean_pnl"] >= 0 else "negative"
        sharpe_cls = "positive" if row["sharpe"] > 0.1 else ("negative" if row["sharpe"] < -0.1 else "neutral")

        label = {"no_hedge": "NO HEDGE", "delta": "DELTA (BS)", "ppo": "PPO AGENT", "sac": "SAC AGENT"}.get(strat, strat.upper())

        with col:
            st.markdown(
                f"""<div class="ws-card">
                  <div class="ws-card-title" style="color:{color};">{label}</div>
                  <div class="ws-card-value {pnl_cls}" style="font-size:22px;">{'+' if row['mean_pnl']>=0 else ''}{row['mean_pnl']:.4f}</div>
                  <div class="ws-card-sub">Mean Terminal P&amp;L</div>
                  <hr style="border-color:#1e3a5f;margin:10px 0 8px;">
                  <div style="display:flex;justify-content:space-between;font-size:11px;">
                    <span style="color:#6b7a8d;">Sharpe Ratio</span>
                    <span class="{sharpe_cls}" style="font-family:'JetBrains Mono',monospace;">{row['sharpe']:+.3f}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;font-size:11px;margin-top:4px;">
                    <span style="color:#6b7a8d;">95% VaR</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:#ff4b4b;">{row['var_95']:+.4f}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;font-size:11px;margin-top:4px;">
                    <span style="color:#6b7a8d;">P&amp;L Std Dev</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:#d4d8e2;">{row['std_pnl']:.4f}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;font-size:11px;margin-top:4px;">
                    <span style="color:#6b7a8d;">Avg TC Paid</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:#6b7a8d;">{row['avg_tc']:.4f}</span>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )


# ── Scenario Performance Table ────────────────────────────────────────────────

if df is not None:
    st.markdown('<div class="ws-section-header">STRATEGY PERFORMANCE ACROSS ALL SCENARIOS (MEAN P&amp;L)</div>',
                unsafe_allow_html=True)

    scenario_labels = {
        "base":          "Base Market",
        "high_tc":       "High Transaction Cost",
        "vol_mismatch":  "Vol Mismatch",
        "regime_switch": "Regime Switching",
    }
    strategies_all = [s for s in ["no_hedge", "delta", "ppo", "sac"] if s in df["strategy"].values]

    table_html = '<table class="ws-table"><thead><tr>'
    table_html += '<th>SCENARIO</th>'
    for s in strategies_all:
        col = STRATEGY_COLOR.get(s, "#d4d8e2")
        label = {"no_hedge": "NO HEDGE", "delta": "DELTA (BS)", "ppo": "PPO AGENT", "sac": "SAC AGENT"}.get(s, s.upper())
        table_html += f'<th style="color:{col};">{label}</th>'
    table_html += '<th style="color:#6b7a8d;">BEST RL WINS BY</th></tr></thead><tbody>'

    for sc, sc_label in scenario_labels.items():
        sub = df[df["scenario"] == sc]
        if sub.empty:
            continue

        row_vals = {}
        for s in strategies_all:
            m = sub[sub["strategy"] == s]
            row_vals[s] = float(m["mean_pnl"].values[0]) if len(m) else None

        delta_pnl = row_vals.get("delta")
        rl_pnls: list[float] = [
            float(row_vals[s]) for s in ["ppo", "sac"] if row_vals.get(s) is not None
        ]
        best_rl = max(rl_pnls) if rl_pnls else None
        advantage = (best_rl - delta_pnl) if (best_rl is not None and delta_pnl is not None) else None

        table_html += f'<tr><td style="color:#6b7a8d;">{sc_label}</td>'
        for s in strategies_all:
            v = row_vals.get(s)
            if v is None:
                table_html += "<td>—</td>"
                continue
            cls = "positive" if v >= 0 else "negative"
            m   = sub[sub["strategy"] == s]
            std = float(m["std_pnl"].values[0]) if len(m) else 0
            table_html += f'<td class="{cls}">{v:+.4f} <span style="font-size:9px;color:#6b7a8d;">[σ={std:.3f}]</span></td>'

        if advantage is not None:
            adv_cls = "positive" if advantage > 0 else "negative"
            table_html += f'<td class="{adv_cls}" style="font-family:\'JetBrains Mono\',monospace;">{advantage:+.4f}</td>'
        else:
            table_html += "<td>—</td>"
        table_html += "</tr>"

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("Positive 'Best RL Wins By' = RL beats delta hedge. RL advantage is strongest in High TC and Vol Mismatch regimes.")


# ── Research Design ───────────────────────────────────────────────────────────

st.markdown('<div class="ws-section-header">RESEARCH DESIGN</div>', unsafe_allow_html=True)

arch_cols = st.columns(4)

with arch_cols[0]:
    st.markdown("""
    <div class="ws-card" style="min-height:220px;">
      <div class="ws-card-title" style="color:#4a9eff;">MARKET ENVIRONMENT</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.8;margin-top:8px;">
        <b>Real Data (Primary)</b><br>
        5 years SPY · 1,200+ windows<br>
        COVID crash · Rate shock<br>
        Bull/bear regimes included<br><br>
        <b>Synthetic (Baseline)</b><br>
        GBM · σ=20% · μ=5%<br>
        Regime switching (15↔35%)<br>
        Vol mismatch scenarios
      </div>
    </div>
    """, unsafe_allow_html=True)

with arch_cols[1]:
    st.markdown("""
    <div class="ws-card" style="min-height:220px;">
      <div class="ws-card-title" style="color:#9b5de5;">STATE SPACE (6 FEATURES)</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.8;margin-top:8px;">
        <span class="ticker">S/S₀</span> — normalized spot<br>
        <span class="ticker">τ/T</span> — time-to-expiry ratio<br>
        <span class="ticker">Δ</span> — BS delta N(d₁)<br>
        <span class="ticker">Γ·S·√τ</span> — gamma exposure<br>
        <span class="ticker">h_t</span> — current holding<br>
        <span class="ticker">log(S/K)</span> — log-moneyness
      </div>
    </div>
    """, unsafe_allow_html=True)

with arch_cols[2]:
    st.markdown("""
    <div class="ws-card" style="min-height:220px;">
      <div class="ws-card-title" style="color:#00d4aa;">RL AGENTS</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.8;margin-top:8px;">
        <b style="color:#00d4aa;">PPO</b> — On-policy<br>
        500k steps · 8 parallel envs<br>
        [256, 256] MLP · clip ε=0.2<br>
        Entropy coef=0.005<br><br>
        <b style="color:#f5c518;">SAC</b> — Off-policy<br>
        300k steps · replay buffer 200k<br>
        [256, 256] MLP · auto entropy<br>
        State-dependent exploration
      </div>
    </div>
    """, unsafe_allow_html=True)

with arch_cols[3]:
    st.markdown("""
    <div class="ws-card" style="min-height:220px;">
      <div class="ws-card-title" style="color:#f5c518;">REWARD FUNCTION</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.8;margin-top:8px;">
        <b>Penalise hedging error &amp; costs:</b><br><br>
        <span class="ticker">r_t = −λ·(ΔV_t)²</span><br>
        <span style="color:#6b7a8d;font-size:11px;">quadratic P&amp;L variance</span><br><br>
        <span class="ticker">− 0.5λ·max(−ΔV_t,0)²</span><br>
        <span style="color:#6b7a8d;font-size:11px;">asymmetric downside penalty</span><br><br>
        <span class="ticker">r_T −= λ_T·V_T²</span><br>
        <span style="color:#6b7a8d;font-size:11px;">terminal settlement (λ_T=5)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Why RL Over Black-Scholes ─────────────────────────────────────────────────

st.markdown('<div class="ws-section-header">WHY REINFORCEMENT LEARNING OVER BLACK-SCHOLES DELTA HEDGING</div>',
            unsafe_allow_html=True)

why_cols = st.columns(2)

with why_cols[0]:
    st.markdown("""
    <div class="ws-card">
      <div class="ws-card-title" style="color:#ff4b4b;">BLACK-SCHOLES DELTA HEDGE — THEORETICAL LIMITATIONS</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.9;margin-top:10px;">
        The classical approach solves for the hedge ratio that makes the portfolio locally
        delta-neutral under the assumption of <b>continuous, costless rebalancing</b> in a
        world where volatility is constant and known.<br><br>
        In practice, every assumption is violated:<br><br>
        <b style="color:#ff4b4b;">Discrete rebalancing</b> — real traders hedge daily or weekly, not continuously.
        Gamma risk accumulates between rebalances and is unaccounted for.<br><br>
        <b style="color:#ff4b4b;">Transaction costs</b> — each stock trade incurs bid-ask spread and
        market impact. Delta hedging ignores this, causing it to overtrade in high-cost environments.<br><br>
        <b style="color:#ff4b4b;">Vol model error</b> — options are priced at implied vol; the actual
        realized vol rarely matches. A delta computed at the wrong vol is systematically wrong.<br><br>
        <b style="color:#ff4b4b;">No regime awareness</b> — the BS delta uses a single σ; it cannot
        adapt when the market enters a crash regime with 3× normal vol.
      </div>
    </div>
    """, unsafe_allow_html=True)

with why_cols[1]:
    st.markdown("""
    <div class="ws-card">
      <div class="ws-card-title" style="color:#00d4aa;">REINFORCEMENT LEARNING HEDGE — HOW IT ADDRESSES THESE</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.9;margin-top:10px;">
        The RL agent learns a <b>policy mapping market state → hedge ratio</b> by interacting
        with thousands of historical market episodes. It receives a reward that directly
        penalises hedging error and implicitly penalises excessive trading through the
        cumulative P&amp;L signal.<br><br>
        <b style="color:#00d4aa;">Discrete rebalancing</b> — the agent trains in discrete daily
        steps, so it learns to account for the gap between rebalances naturally.<br><br>
        <b style="color:#00d4aa;">Transaction cost awareness</b> — because TC reduces P&amp;L which
        reduces reward, the agent is incentivised to rebalance only when the hedge benefit
        exceeds the cost. This emergent property is not programmed in.<br><br>
        <b style="color:#00d4aa;">Regime adaptation</b> — with 5 years of training data including
        the 2020 crash, 2022 rate shock, and multiple vol regimes, the agent learns which
        state features signal a regime change and adjusts aggressively.<br><br>
        <b style="color:#00d4aa;">No model assumptions</b> — the reward function contains no
        reference to any σ or μ. The agent infers what it needs from price dynamics alone.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Navigation ────────────────────────────────────────────────────────────────

st.markdown('<div class="ws-section-header">NAVIGATE THE PLATFORM</div>', unsafe_allow_html=True)

nav_cols = st.columns(5)
pages = [
    ("01", "Live Demo",    "4a9eff",
     "Watch any agent hedge a single option episode frame-by-frame with animated P&L, delta tracking, and hedge error charts. Compare PPO/SAC vs the Black-Scholes baseline side-by-side."),
    ("02", "Training",     "00d4aa",
     "Launch or continue PPO/SAC training on real SPY data. Monitor live reward curves, view training statistics, and inspect hyperparameter configuration."),
    ("03", "Evaluation",   "9b5de5",
     "Full 5-strategy × 4-scenario performance dashboard. Sharpe ratios, VaR/CVaR tail risk, transaction cost analysis, and P&L distribution curves."),
    ("04", "Scenario Lab", "f5c518",
     "Design custom market scenarios — adjust vol, transaction costs, drift, and regime parameters. Run Monte Carlo comparisons and explore when RL outperforms classical methods."),
    ("05", "Market Data",  "00d4aa",
     "Live SPY price feed refreshing every second. SPY options chain, implied vol surface, 5-year price history, and vol regime classification with real market data."),
]
for col, (num, name, color, desc) in zip(nav_cols, pages):
    with col:
        st.markdown(
            f"""<div class="ws-card" style="min-height:180px;">
              <div style="font-size:10px;letter-spacing:3px;color:#1e3a5f;
                   font-family:'JetBrains Mono',monospace;margin-bottom:6px;">{num}</div>
              <div style="font-size:13px;font-weight:600;color:#{color};margin:4px 0 10px;">{name}</div>
              <div style="font-size:11px;color:#6b7a8d;line-height:1.6;">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Technical Specification ───────────────────────────────────────────────────

st.markdown('<div class="ws-section-header">TECHNICAL SPECIFICATION</div>', unsafe_allow_html=True)

spec_cols = st.columns(3)

with spec_cols[0]:
    st.markdown("""
    <div class="ws-card">
      <div class="ws-card-title" style="color:#4a9eff;">DATA PIPELINE</div>
      <table class="ws-table" style="margin-top:8px;">
        <tr><td style="color:#6b7a8d;">Source</td><td>Yahoo Finance (yfinance)</td></tr>
        <tr><td style="color:#6b7a8d;">Ticker</td><td>SPY (S&amp;P 500 ETF)</td></tr>
        <tr><td style="color:#6b7a8d;">History</td><td>5 years daily OHLCV</td></tr>
        <tr><td style="color:#6b7a8d;">Training windows</td><td>1,200+ (30-day)</td></tr>
        <tr><td style="color:#6b7a8d;">Calibrated σ</td><td>17.1% annualized</td></tr>
        <tr><td style="color:#6b7a8d;">Options chain</td><td>Live SPY calls (4 expiries)</td></tr>
        <tr><td style="color:#6b7a8d;">Vol surface</td><td>Strike × maturity IV grid</td></tr>
        <tr><td style="color:#6b7a8d;">Regime labels</td><td>Median RVol threshold</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

with spec_cols[1]:
    st.markdown("""
    <div class="ws-card">
      <div class="ws-card-title" style="color:#00d4aa;">EVALUATION METRICS</div>
      <table class="ws-table" style="margin-top:8px;">
        <tr><td style="color:#6b7a8d;">Sharpe Ratio</td><td>Mean P&amp;L / Std P&amp;L</td></tr>
        <tr><td style="color:#6b7a8d;">95% VaR</td><td>5th percentile P&amp;L</td></tr>
        <tr><td style="color:#6b7a8d;">95% CVaR</td><td>Expected shortfall</td></tr>
        <tr><td style="color:#6b7a8d;">Max Drawdown</td><td>Worst single episode P&amp;L</td></tr>
        <tr><td style="color:#6b7a8d;">% Loss</td><td>Fraction negative P&amp;L episodes</td></tr>
        <tr><td style="color:#6b7a8d;">Avg TC</td><td>Mean cumulative transaction cost</td></tr>
        <tr><td style="color:#6b7a8d;">Rebalances</td><td>Number of hedge adjustments</td></tr>
        <tr><td style="color:#6b7a8d;">Hedge Error σ</td><td>Std dev of step-level ΔV</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

with spec_cols[2]:
    st.markdown("""
    <div class="ws-card">
      <div class="ws-card-title" style="color:#9b5de5;">TECH STACK</div>
      <table class="ws-table" style="margin-top:8px;">
        <tr><td style="color:#6b7a8d;">RL Framework</td><td>Stable-Baselines3</td></tr>
        <tr><td style="color:#6b7a8d;">Environment</td><td>Gymnasium (OpenAI)</td></tr>
        <tr><td style="color:#6b7a8d;">UI</td><td>Streamlit + Plotly</td></tr>
        <tr><td style="color:#6b7a8d;">Market data</td><td>yfinance</td></tr>
        <tr><td style="color:#6b7a8d;">Option pricing</td><td>Black-Scholes (scipy)</td></tr>
        <tr><td style="color:#6b7a8d;">Language</td><td>Python 3.13</td></tr>
        <tr><td style="color:#6b7a8d;">Neural nets</td><td>PyTorch (via SB3)</td></tr>
        <tr><td style="color:#6b7a8d;">Training log</td><td>TensorBoard</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)
