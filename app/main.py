"""RL Derivative Hedging"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np

from components.charts import inject_css, fmt_pnl, fmt_sharpe, STRATEGY_COLOR  # noqa: E402

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="RL Hedging | Trading Desk",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(inject_css(), unsafe_allow_html=True)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_CSV = os.path.join(ROOT, "results", "evaluation_results.csv")
MODEL_DIR   = os.path.join(ROOT, "models")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _model_status(name: str) -> tuple[bool, str]:
    exists = os.path.exists(os.path.join(MODEL_DIR, f"{name}_hedger.zip"))
    badge  = '<span class="badge-live">● LIVE</span>' if exists else '<span class="badge-off">○ NOT TRAINED</span>'
    return exists, badge


def _load_results() -> pd.DataFrame | None:
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV)
    return None


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## RL HEDGING DESK")
    st.markdown('<hr class="ws-divider">', unsafe_allow_html=True)

    ppo_ok, ppo_badge = _model_status("ppo")
    sac_ok, sac_badge = _model_status("sac")
    results_ok = os.path.exists(RESULTS_CSV)

    st.markdown("**System Status**")
    st.markdown(f"PPO Agent &nbsp;&nbsp; {ppo_badge}", unsafe_allow_html=True)
    st.markdown(f"SAC Agent &nbsp;&nbsp; {sac_badge}", unsafe_allow_html=True)
    eval_badge = '<span class="badge-live">● READY</span>' if results_ok else '<span class="badge-warn">◐ PENDING</span>'
    st.markdown(f"Eval Data &nbsp;&nbsp; {eval_badge}", unsafe_allow_html=True)

    st.markdown('<hr class="ws-divider">', unsafe_allow_html=True)
    st.markdown("**Quick Start**")
    st.code("# From project root:\nstreamlit run app/main.py", language="bash")

    st.markdown('<hr class="ws-divider">', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:10px;color:#4a5568;">RL DERIVATIVE HEDGING · RESEARCH SYSTEM</span>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown('<div class="ws-title">RL DERIVATIVE HEDGING DESK</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ws-subtitle">Reinforcement Learning — Dynamic Option Hedging · PPO · SAC · Black-Scholes Baselines</div>',
        unsafe_allow_html=True,
    )
with col_time:
    import datetime
    now = datetime.datetime.now()
    st.markdown(
        f'<div style="text-align:right;padding-top:10px;">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#6b7a8d;">'
        f'{now.strftime("%Y-%m-%d %H:%M:%S")}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="ws-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Key Metrics Strip
# ─────────────────────────────────────────────

df = _load_results()

if df is not None:
    st.markdown('<div class="ws-section-header">KEY PERFORMANCE INDICATORS — BASE SCENARIO</div>', unsafe_allow_html=True)
    base = df[df["scenario"] == "base"]

    strategies_show = [s for s in ["no_hedge", "delta", "ppo", "sac"] if s in base["strategy"].values]
    cols = st.columns(len(strategies_show))

    for col, strat in zip(cols, strategies_show):
        _rows = base[base["strategy"] == strat]
        if _rows.empty:
            continue
        row = _rows.iloc[0]
        color = STRATEGY_COLOR.get(strat, "#d4d8e2")
        pnl_cls = "positive" if row["mean_pnl"] >= 0 else "negative"
        sharpe_cls = "positive" if row["sharpe"] > 0.1 else ("negative" if row["sharpe"] < -0.1 else "neutral")

        with col:
            st.markdown(
                f"""<div class="ws-card">
                  <div class="ws-card-title" style="color:{color};">{strat.upper()}</div>
                  <div class="ws-card-value {pnl_cls}">{'+' if row['mean_pnl']>=0 else ''}{row['mean_pnl']:.4f}</div>
                  <div class="ws-card-sub">Mean P&amp;L</div>
                  <hr style="border-color:#1e3a5f;margin:10px 0 8px;">
                  <div style="display:flex;justify-content:space-between;font-size:11px;">
                    <span style="color:#6b7a8d;">Sharpe</span>
                    <span class="{sharpe_cls}" style="font-family:'JetBrains Mono',monospace;">{row['sharpe']:+.3f}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;font-size:11px;margin-top:4px;">
                    <span style="color:#6b7a8d;">VaR 95%</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:#ff4b4b;">{row['var_95']:+.4f}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;font-size:11px;margin-top:4px;">
                    <span style="color:#6b7a8d;">Std Dev</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:#d4d8e2;">{row['std_pnl']:.4f}</span>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
else:
    st.info("No evaluation results found. Run the pipeline to generate data: `python run_pipeline.py`")


# ─────────────────────────────────────────────
# Scenario Performance Table
# ─────────────────────────────────────────────

if df is not None:
    st.markdown('<div class="ws-section-header">STRATEGY PERFORMANCE ACROSS SCENARIOS</div>', unsafe_allow_html=True)

    scenario_labels = {
        "base": "Base Market", "high_tc": "High TC",
        "vol_mismatch": "Vol Mismatch", "regime_switch": "Regime Switch",
    }

    # Build pivot-style HTML table
    strategies_all = [s for s in ["no_hedge", "delta", "ppo", "sac"] if s in df["strategy"].values]
    table_html = '<table class="ws-table"><thead><tr>'
    table_html += '<th>SCENARIO</th>'
    for s in strategies_all:
        col = STRATEGY_COLOR.get(s, "#d4d8e2")
        table_html += f'<th style="color:{col};">{s.upper()}</th>'
    table_html += '</tr></thead><tbody>'

    for sc, sc_label in scenario_labels.items():
        sub = df[df["scenario"] == sc]
        if sub.empty:
            continue
        table_html += f'<tr><td style="color:#6b7a8d;">{sc_label}</td>'
        for s in strategies_all:
            matching = sub[sub["strategy"] == s]
            if matching.empty:
                table_html += "<td>—</td>"
                continue
            r = matching.iloc[0]
            cls = "positive" if r["mean_pnl"] >= 0 else "negative"
            table_html += (
                f'<td class="{cls}">{r["mean_pnl"]:+.4f} '
                f'<span style="font-size:9px;color:#6b7a8d;">'
                f'[σ={r["std_pnl"]:.3f}]</span></td>'
            )
        table_html += "</tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Architecture Overview
# ─────────────────────────────────────────────

st.markdown('<div class="ws-section-header">SYSTEM ARCHITECTURE</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="ws-card">
      <div class="ws-card-title" style="color:#4a9eff;">MARKET SIMULATION</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.7;">
        <span class="ticker">GBM</span> — Geometric Brownian Motion<br>
        <span class="ticker">σ = 0.20</span> annualised vol<br>
        <span class="ticker">μ = 0.05</span> drift<br>
        <span class="ticker">dt = 1/252</span> daily steps<br><br>
        <b style="color:#6b7a8d;">Extensions:</b><br>
        Regime-switching (15% ↔ 35%)<br>
        Volatility mismatch<br>
        Transaction costs
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="ws-card">
      <div class="ws-card-title" style="color:#9b5de5;">OPTION PRICING</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.7;">
        <b>Black-Scholes European Call</b><br><br>
        <span class="ticker">K = 100</span> strike<br>
        <span class="ticker">T = 30d</span> maturity<br>
        <span class="ticker">r = 0.01</span> risk-free rate<br><br>
        <b style="color:#6b7a8d;">Greeks computed:</b><br>
        Δ (delta) — hedge ratio<br>
        Γ (gamma) — convexity risk
      </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="ws-card">
      <div class="ws-card-title" style="color:#00d4aa;">RL AGENTS</div>
      <div style="font-size:12px;color:#d4d8e2;line-height:1.7;">
        <b>PPO</b> — Proximal Policy Optimisation<br>
        500k steps · 8 parallel envs<br>
        Net: [256, 256] MLP<br><br>
        <b>SAC</b> — Soft Actor-Critic<br>
        300k steps · entropy auto-tune<br>
        Net: [256, 256] MLP<br><br>
        <b style="color:#6b7a8d;">Reward:</b> −(ΔV)² − risk penalty
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Navigation Guide
# ─────────────────────────────────────────────

st.markdown('<div class="ws-section-header">NAVIGATION</div>', unsafe_allow_html=True)

nav_cols = st.columns(4)
pages = [
    ("01", "Live Demo",    "Watch an agent hedge an option episode in real-time with animated charts."),
    ("02", "Training",     "Monitor training progress, view reward curves, and inspect model performance."),
    ("03", "Evaluation",   "Full strategy comparison across all scenarios with interactive metrics."),
    ("04", "Scenario Lab", "Adjust volatility, costs, and regime parameters and compare strategies live."),
]
for col, (num, name, desc) in zip(nav_cols, pages):
    with col:
        st.markdown(
            f"""<div class="ws-card" style="min-height:110px;">
              <div style="font-size:10px;letter-spacing:3px;color:#1e3a5f;font-family:'JetBrains Mono',monospace;margin-bottom:4px;">{num}</div>
              <div style="font-size:13px;font-weight:600;color:#d4d8e2;margin:4px 0;">{name}</div>
              <div style="font-size:11px;color:#6b7a8d;">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )
