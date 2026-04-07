"""Live Demo — watch the agent hedge step-by-step."""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd

from components.charts import inject_css, live_episode_chart, STRATEGY_COLOR, BG2, TEXT, GREEN, GOLD, RED  # noqa
from components.runner import collect_episode, load_model, ENV_PRESETS  # noqa

st.set_page_config(page_title="Live Demo | RL Hedging", page_icon=None, layout="wide")
st.markdown(inject_css(), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## LIVE DEMO CONTROLS")
    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

    scenario_name = st.selectbox(
        "Market Scenario", list(ENV_PRESETS.keys()),
        help="The market environment the agent hedges in",
    )
    strategy = st.selectbox(
        "Strategy",
        ["delta", "ppo", "sac", "no_hedge", "random"],
        format_func=lambda x: {
            "delta":    "Delta Hedge (Black-Scholes)",
            "ppo":      "PPO Agent",
            "sac":      "SAC Agent",
            "no_hedge": "No Hedge",
            "random":   "Random",
        }.get(x, x),
    )
    seed = st.slider("Episode Seed", 0, 200, 42, help="Different seeds = different price paths")
    speed = st.select_slider(
        "Animation Speed",
        options=["Slow", "Normal", "Fast", "Instant"],
        value="Normal",
    )
    SPEED_MAP = {"Slow": 0.15, "Normal": 0.07, "Fast": 0.02, "Instant": 0.0}
    delay = SPEED_MAP[speed]

    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

    compare_mode = st.checkbox("Compare vs Delta Hedge", value=True)

    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

    # Show scenario info
    env_kw = ENV_PRESETS[scenario_name]
    st.markdown(f"""
    **Scenario Parameters**
    | Param | Value |
    |---|---|
    | Spot | {env_kw['s0']} |
    | Strike | {env_kw['strike']} |
    | Vol (model) | {env_kw['sigma']:.0%} |
    | Realized Vol | {env_kw.get('realized_sigma', env_kw['sigma']):.0%} |
    | TC Rate | {env_kw['transaction_cost']:.2%} |
    | Maturity | 30 days |
    | Regime Switch | {'Yes' if env_kw.get('regime_switching') else 'No'} |
    """)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown(
    f'<div class="ws-title">LIVE HEDGING DEMO</div>'
    f'<div class="ws-subtitle">{scenario_name} · Strategy: {strategy.upper()} · Seed #{seed}</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Run episode
# ─────────────────────────────────────────────

run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("▶  RUN EPISODE", type="primary", width="stretch")

if "episode_df"  not in st.session_state:
    st.session_state.episode_df = None
if "compare_df"  not in st.session_state:
    st.session_state.compare_df = None

model = None
if strategy in ("ppo", "sac"):
    model = load_model(strategy)
    if model is None:
        st.warning(f"{strategy.upper()} model not found. Train it first on the Training page.")
        st.stop()

if run_btn:
    env_kw = ENV_PRESETS[scenario_name]

    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()

    full_df = collect_episode(env_kw, strategy, model=model, seed=seed)
    if full_df is None or len(full_df) < 2:
        st.error("Episode failed to produce data. Check that the environment initialises correctly.")
        st.stop()
    compare_df = collect_episode(env_kw, "delta", model=None, seed=seed) if compare_mode else None
    if compare_df is not None and len(compare_df) < 2:
        compare_df = None  # silently drop failed comparison rather than crash

    st.session_state.episode_df = full_df
    st.session_state.compare_df = compare_df

    if delay > 0:
        for i in range(2, len(full_df) + 1):
            partial = full_df.iloc[:i]
            with chart_placeholder:
                fig = live_episode_chart(partial, strategy)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            time.sleep(delay)
    else:
        with chart_placeholder:
            fig = live_episode_chart(full_df, strategy)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

elif st.session_state.episode_df is not None:
    fig = live_episode_chart(st.session_state.episode_df, strategy)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
else:
    st.markdown(
        '<div style="text-align:center;padding:60px;color:#4a5568;">'
        '<div style="font-size:11px;letter-spacing:3px;color:#1e3a5f;font-family:\'JetBrains Mono\',monospace;">READY</div>'
        '<div style="font-size:14px;margin-top:12px;color:#4a5568;">Press RUN EPISODE to start the live hedging demo</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Episode summary metrics
# ─────────────────────────────────────────────

if st.session_state.episode_df is not None:
    df = st.session_state.episode_df
    cdf = st.session_state.compare_df

    if len(df) < 2:
        st.warning("Episode data incomplete — run again.")
        st.stop()

    st.markdown('<div class="ws-section-header">EPISODE SUMMARY</div>', unsafe_allow_html=True)

    final_pv  = float(df["portfolio_value"].iloc[-1])
    total_tc  = float(df["cumulative_tc"].iloc[-1])
    max_draw  = float(df["portfolio_value"].min())
    n_trades  = int((df["hedge_position"].diff().abs() > 0.01).sum())
    hedge_err = float(df["portfolio_value"].std())

    cols = st.columns(5)
    metrics = [
        ("FINAL P&L",         f"{final_pv:+.4f}",  "positive" if final_pv >= 0 else "negative"),
        ("TOTAL TC PAID",      f"{total_tc:.4f}",   "negative"),
        ("MAX DRAWDOWN",       f"{max_draw:+.4f}",  "positive" if max_draw >= 0 else "negative"),
        ("REBALANCES",         str(n_trades),        "neutral"),
        ("HEDGE ERROR (σ)",    f"{hedge_err:.4f}",  "neutral"),
    ]
    for col, (title, val, cls) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="ws-card">'
                f'<div class="ws-card-title">{title}</div>'
                f'<div class="ws-card-value {cls}">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Side-by-side comparison with delta if available
    if cdf is not None:
        st.markdown('<div class="ws-section-header">COMPARISON vs DELTA HEDGE</div>', unsafe_allow_html=True)

        delta_pv = float(cdf["portfolio_value"].iloc[-1])
        delta_tc = float(cdf["cumulative_tc"].iloc[-1])

        comp_cols = st.columns(2)
        with comp_cols[0]:
            diff = final_pv - delta_pv
            cls = "positive" if diff >= 0 else "negative"
            st.markdown(
                f'<div class="ws-card">'
                f'<div class="ws-card-title">P&L ADVANTAGE vs DELTA</div>'
                f'<div class="ws-card-value {cls}">{diff:+.4f}</div>'
                f'<div class="ws-card-sub">RL: {final_pv:+.4f} · Delta: {delta_pv:+.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with comp_cols[1]:
            tc_saved = delta_tc - total_tc
            cls = "positive" if tc_saved >= 0 else "negative"
            st.markdown(
                f'<div class="ws-card">'
                f'<div class="ws-card-title">TRANSACTION COST SAVING</div>'
                f'<div class="ws-card-value {cls}">{tc_saved:+.4f}</div>'
                f'<div class="ws-card-sub">RL: {total_tc:.4f} · Delta: {delta_tc:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Raw step data expander
    with st.expander("Raw Episode Data", expanded=False):
        display_df = df[["step","spot","delta","hedge_position","portfolio_value","delta_v","cumulative_tc","realized_vol"]].copy()
        display_df = display_df.round(4)
        st.dataframe(display_df, width="stretch", height=300)
