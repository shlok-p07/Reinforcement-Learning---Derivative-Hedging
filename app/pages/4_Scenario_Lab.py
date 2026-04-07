"""Scenario Lab — interactive parameter exploration."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts import (  # noqa
    inject_css, pnl_distribution, STRATEGY_COLOR, BG2, GREEN, RED, GOLD, BLUE, GREY, _LAYOUT,
)
from components.runner import collect_episode, load_model  # noqa

st.set_page_config(page_title="Scenario Lab | RL Hedging", page_icon=None, layout="wide")
st.markdown(inject_css(), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar — parameter controls
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## SCENARIO LAB")
    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

    st.markdown("**Market Parameters**")
    sigma_model    = st.slider("Model Volatility (σ)", 0.05, 0.60, 0.20, step=0.01, format="%.2f")
    sigma_realized = st.slider("Realised Volatility (σ_real)", 0.05, 0.60, 0.20, step=0.01, format="%.2f")
    tc_rate        = st.slider("Transaction Cost Rate", 0.0, 0.02, 0.001, step=0.001, format="%.3f")
    spot0          = st.slider("Initial Spot (S₀)", 80, 120, 100, step=5)
    strike         = st.slider("Strike (K)", 80, 120, 100, step=5)
    maturity_days  = st.slider("Maturity (days)", 5, 63, 30, step=5)

    st.markdown("**Regime Switching**")
    regime_on   = st.checkbox("Enable Regime Switching", value=False)
    sigma_low   = st.slider("Low Vol Regime (σ_low)",  0.05, 0.40, 0.15, step=0.01, disabled=not regime_on)
    sigma_high  = st.slider("High Vol Regime (σ_high)", 0.10, 0.80, 0.35, step=0.01, disabled=not regime_on)

    st.markdown("**Simulation**")
    n_episodes = st.slider("Monte Carlo Episodes", 50, 300, 100, step=50)
    strategies = st.multiselect(
        "Strategies to Compare",
        ["no_hedge", "delta", "ppo", "sac", "random"],
        default=["no_hedge", "delta"],
        format_func=lambda x: {"no_hedge":"No Hedge","delta":"Delta","ppo":"PPO","sac":"SAC","random":"Random"}.get(x,x),
    )

    run_btn = st.button("▶  RUN COMPARISON", type="primary", width="stretch")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<div class="ws-title">SCENARIO LABORATORY</div>', unsafe_allow_html=True)
st.markdown('<div class="ws-subtitle">Adjust market parameters and compare strategies in real time</div>', unsafe_allow_html=True)
st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

# Show current scenario config as a ticker-style strip
vol_mismatch = abs(sigma_model - sigma_realized) > 0.005
config_html = (
    f'<div class="ws-card" style="display:flex;gap:30px;align-items:center;padding:12px 20px;">'
    f'<span class="ticker">S₀={spot0}</span>'
    f'<span class="ticker">K={strike}</span>'
    f'<span class="ticker" style="color:#00d4aa;">σ_model={sigma_model:.0%}</span>'
    f'<span class="ticker" style="color:{"#f5c518" if vol_mismatch else "#00d4aa"};">σ_real={sigma_realized:.0%}</span>'
    f'<span class="ticker" style="color:#ff4b4b;">TC={tc_rate:.2%}</span>'
    f'<span class="ticker">T={maturity_days}d</span>'
    f'<span class="ticker">Regime={"ON" if regime_on else "OFF"}</span>'
)
if vol_mismatch:
    config_html += f'<span class="badge-warn">VOL MISMATCH +{(sigma_realized-sigma_model)*100:+.0f}%</span>'
if regime_on:
    config_html += f'<span class="badge-warn">REGIME SWITCHING</span>'
config_html += "</div>"
st.markdown(config_html, unsafe_allow_html=True)

st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────

if "lab_results" not in st.session_state:
    st.session_state.lab_results = None

if run_btn and strategies:
    env_kwargs = dict(
        s0=float(spot0), mu=0.05, sigma=sigma_model,
        dt=1/252, maturity=maturity_days/252,
        strike=float(strike), rate=0.01,
        transaction_cost=tc_rate,
        realized_sigma=sigma_realized if not regime_on else None,
        regime_switching=regime_on,
        sigma_low=sigma_low if regime_on else 0.15,
        sigma_high=sigma_high if regime_on else 0.35,
    )
    # Remove None values
    env_kwargs = {k: v for k, v in env_kwargs.items() if v is not None}

    models = {}
    for s in strategies:
        if s in ("ppo", "sac"):
            m = load_model(s)
            if m is None:
                st.warning(f"{s.upper()} model not found — skipping.")
            else:
                models[s] = m

    with st.spinner(f"Simulating {n_episodes} episodes × {len(strategies)} strategies…"):
        results: dict[str, dict] = {}
        prog = st.progress(0)
        valid_strategies = [s for s in strategies if s not in ("ppo","sac") or s in models]

        for si, strat in enumerate(valid_strategies):
            pnls, tcs = [], []
            for ep in range(n_episodes):
                ep_df = collect_episode(env_kwargs, strat, model=models.get(strat), seed=ep)
                if ep_df is None or len(ep_df) < 2:
                    continue
                pnls.append(float(ep_df["portfolio_value"].iloc[-1]))
                tcs.append(float(ep_df["cumulative_tc"].iloc[-1]))
            if not pnls:
                st.warning(f"{strat.upper()} produced no valid episodes — skipping.")
                prog.progress((si + 1) / len(valid_strategies))
                continue
            pnl_arr = np.array(pnls)
            std = float(np.std(pnl_arr))
            var95 = float(np.percentile(pnl_arr, 5))
            results[strat] = {
                "pnls":      pnl_arr,
                "mean_pnl":  float(np.mean(pnl_arr)),
                "std_pnl":   std,
                "sharpe":    float(np.mean(pnl_arr)/std) if std > 1e-9 else 0.0,
                "var_95":    var95,
                "cvar_95":   float(np.mean(pnl_arr[pnl_arr <= var95])) if (pnl_arr <= var95).any() else float(var95),
                "avg_tc":    float(np.mean(tcs)),
                "pct_loss":  float(np.mean(pnl_arr < 0)),
            }
            prog.progress((si + 1) / len(valid_strategies))

    st.session_state.lab_results = results
    st.success(f"Done! {n_episodes} episodes per strategy.")


# ─────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────

if st.session_state.lab_results:
    res = st.session_state.lab_results
    strats = list(res.keys())

    # Metric cards
    st.markdown('<div class="ws-section-header">RESULTS</div>', unsafe_allow_html=True)
    metric_cols = st.columns(len(strats))

    for col, strat in zip(metric_cols, strats):
        r = res[strat]
        color = STRATEGY_COLOR.get(strat, GREY)
        pnl_cls = "positive" if r["mean_pnl"] >= 0 else "negative"
        sharpe_cls = "positive" if r["sharpe"] > 0.1 else ("negative" if r["sharpe"] < -0.1 else "neutral")
        with col:
            st.markdown(
                f"""<div class="ws-card">
                  <div class="ws-card-title" style="color:{color};">{strat.upper()}</div>
                  <div class="ws-card-value {pnl_cls}">{r['mean_pnl']:+.4f}</div>
                  <div class="ws-card-sub">Mean P&amp;L</div>
                  <hr style="border-color:#1e3a5f;margin:8px 0;">
                  <div style="font-size:11px;display:flex;justify-content:space-between;">
                    <span style="color:#6b7a8d;">Sharpe</span>
                    <span class="{sharpe_cls}" style="font-family:'JetBrains Mono',monospace;">{r['sharpe']:+.3f}</span>
                  </div>
                  <div style="font-size:11px;display:flex;justify-content:space-between;margin-top:4px;">
                    <span style="color:#6b7a8d;">VaR 95%</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:#ff4b4b;">{r['var_95']:+.4f}</span>
                  </div>
                  <div style="font-size:11px;display:flex;justify-content:space-between;margin-top:4px;">
                    <span style="color:#6b7a8d;">Avg TC</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:#d4d8e2;">{r['avg_tc']:.4f}</span>
                  </div>
                  <div style="font-size:11px;display:flex;justify-content:space-between;margin-top:4px;">
                    <span style="color:#6b7a8d;">% Loss eps</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:#d4d8e2;">{r['pct_loss']:.1%}</span>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Distribution chart
    st.markdown('<div class="ws-section-header">P&L DISTRIBUTIONS</div>', unsafe_allow_html=True)
    pnl_dict = {s: r["pnls"] for s, r in res.items()}
    st.plotly_chart(pnl_distribution(pnl_dict), width="stretch", config={"displayModeBar": False})

    # Sensitivity insight
    if "delta" in res and len([s for s in strats if s not in ("delta", "no_hedge", "random")]) > 0:
        rl_strats = [s for s in strats if s in ("ppo", "sac")]
        if rl_strats:
            best_rl = max(rl_strats, key=lambda s: res[s]["sharpe"])
            delta_sharpe = res["delta"]["sharpe"]
            rl_sharpe    = res[best_rl]["sharpe"]
            advantage    = rl_sharpe - delta_sharpe

            cls = "positive" if advantage > 0 else "negative"
            st.markdown(
                f"""<div class="ws-card" style="margin-top:16px;">
                  <div class="ws-card-title" style="color:#00d4aa;">ANALYSIS — RL vs DELTA HEDGE</div>
                  <div style="font-size:13px;color:#d4d8e2;margin-top:8px;">
                    Best RL agent: <b style="color:{STRATEGY_COLOR.get(best_rl, GREEN)};">{best_rl.upper()}</b>
                    &nbsp;|&nbsp;
                    Sharpe advantage: <span class="{cls}" style="font-family:JetBrains Mono,monospace;">{advantage:+.3f}</span>
                    &nbsp;|&nbsp;
                    RL Sharpe: <b>{rl_sharpe:+.3f}</b> &nbsp; Delta Sharpe: <b>{delta_sharpe:+.3f}</b>
                  </div>
                  <div style="font-size:11px;color:#6b7a8d;margin-top:8px;">
                    {'RL outperforms delta hedge in this scenario.' if advantage > 0
                     else 'Delta hedge leads in this scenario — try high TC or vol mismatch settings.'}
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        """<div style="text-align:center;padding:80px;color:#4a5568;">
          <div style="font-size:11px;letter-spacing:3px;color:#1e3a5f;font-family:'JetBrains Mono',monospace;">READY</div>
          <div style="font-size:14px;margin-top:14px;">
            Adjust the parameters in the sidebar and press <b style="color:#00d4aa;">RUN COMPARISON</b>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
