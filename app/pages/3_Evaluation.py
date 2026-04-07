"""Full Strategy Evaluation Dashboard."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np

from components.charts import (  # noqa
    inject_css, sharpe_bars, risk_return_scatter, var_cvar_bars,
    tc_comparison, pnl_distribution, fmt_pnl, fmt_sharpe, STRATEGY_COLOR, SCENARIO_LABEL,
)
from components.runner import collect_episode, load_model, ENV_PRESETS  # noqa

st.set_page_config(page_title="Evaluation | RL Hedging", page_icon=None, layout="wide")
st.markdown(inject_css(), unsafe_allow_html=True)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_CSV = os.path.join(ROOT, "results", "evaluation_results.csv")


def _load_df() -> pd.DataFrame | None:
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV)
    return None


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## EVALUATION CONTROLS")
    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

    n_episodes = st.slider("Episodes per combo", 50, 500, 200, step=50)
    run_eval = st.button("Re-run Evaluation", type="primary", width="stretch")

    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)
    st.markdown("**Filter**")
    df_full = _load_df()
    all_scenarios = list(SCENARIO_LABEL.keys()) if df_full is None else list(df_full["scenario"].unique())
    selected_scenario = st.selectbox("Scenario Focus", ["All"] + all_scenarios)

    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)
    if df_full is not None and st.button("Download CSV", width="stretch"):
        st.download_button(
            "Download evaluation_results.csv",
            df_full.to_csv(index=False),
            file_name="evaluation_results.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────
# Re-run evaluation inline
# ─────────────────────────────────────────────

if run_eval:
    with st.spinner(f"Running evaluation ({n_episodes} episodes per strategy × scenario)…"):
        import subprocess
        env_map = {
            "base":          dict(s0=100, mu=0.05, sigma=0.20, dt=1/252, maturity=30/252, strike=100, rate=0.01, transaction_cost=0.001),
            "high_tc":       dict(s0=100, mu=0.05, sigma=0.20, dt=1/252, maturity=30/252, strike=100, rate=0.01, transaction_cost=0.01),
            "vol_mismatch":  dict(s0=100, mu=0.05, sigma=0.20, dt=1/252, maturity=30/252, strike=100, rate=0.01, transaction_cost=0.001, realized_sigma=0.30),
            "regime_switch": dict(s0=100, mu=0.05, sigma=0.20, dt=1/252, maturity=30/252, strike=100, rate=0.01, transaction_cost=0.001, regime_switching=True, sigma_low=0.15, sigma_high=0.35),
        }

        models = {n: load_model(n) for n in ["ppo", "sac"]}
        rl_available = [n for n, m in models.items() if m is not None]
        if not rl_available:
            st.warning("No trained RL models found — running baselines only. Train PPO/SAC on the Training page first.")
        rows = []
        prog = st.progress(0)
        combo_list = [(sc, st_) for sc in env_map for st_ in (["no_hedge", "delta", "random"] + rl_available)]
        total = len(combo_list)

        for idx, (sc, strat) in enumerate(combo_list):
            pnls, tcs = [], []
            for ep in range(n_episodes):
                ep_df = collect_episode(env_map[sc], strat, model=models.get(strat), seed=ep)
                pnls.append(float(ep_df["portfolio_value"].iloc[-1]))
                tcs.append(float(ep_df["cumulative_tc"].iloc[-1]))
            pnl_arr = np.array(pnls)
            var95 = float(np.percentile(pnl_arr, 5))
            std   = float(np.std(pnl_arr))
            rows.append({
                "strategy": strat, "scenario": sc,
                "mean_pnl":  float(np.mean(pnl_arr)),
                "std_pnl":   std,
                "sharpe":    float(np.mean(pnl_arr)/std) if std > 1e-9 else 0.0,
                "var_95":    var95,
                "cvar_95":   float(np.mean(pnl_arr[pnl_arr <= var95])) if (pnl_arr <= var95).any() else var95,
                "max_loss":  float(np.min(pnl_arr)),
                "pct_loss":  float(np.mean(pnl_arr < 0)),
                "avg_tc":    float(np.mean(tcs)),
            })
            prog.progress((idx + 1) / total)

        df_new = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
        df_new.to_csv(RESULTS_CSV, index=False)
        st.success(f"Evaluation complete — {len(rows)} results saved.")
        st.rerun()


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<div class="ws-title">STRATEGY EVALUATION DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<div class="ws-subtitle">5 strategies × 4 market scenarios × 8 risk & performance metrics</div>', unsafe_allow_html=True)
st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

df = _load_df()
if df is None:
    st.markdown(
        """<div class="ws-card" style="text-align:center;padding:50px;">
          <div style="font-size:11px;letter-spacing:3px;color:#1e3a5f;font-family:'JetBrains Mono',monospace;">NO DATA</div>
          <div style="font-size:14px;color:#6b7a8d;margin-top:14px;">
            No evaluation results yet.<br>
            Click <b style="color:#00d4aa;">Re-run Evaluation</b> in the sidebar, or run:
          </div>
          <code style="font-size:12px;color:#f5c518;">python run_pipeline.py --eval-only --n-episodes 100</code>
        </div>""",
        unsafe_allow_html=True,
    )
    st.stop()

# Filter
if selected_scenario != "All":
    df_view = df[df["scenario"] == selected_scenario]
else:
    df_view = df.copy()


# ─────────────────────────────────────────────
# Full results table
# ─────────────────────────────────────────────

st.markdown('<div class="ws-section-header">FULL RESULTS TABLE</div>', unsafe_allow_html=True)

display_df = df_view.copy()
display_df["scenario"] = display_df["scenario"].map(SCENARIO_LABEL).fillna(display_df["scenario"])

# Style the DataFrame
def _color_pnl(v):
    try:
        fv = float(v)
        return f"color: {'#00d4aa' if fv >= 0 else '#ff4b4b'}"
    except Exception:
        return ""

styled = (
    display_df
    .rename(columns={
        "strategy":"Strategy","scenario":"Scenario",
        "mean_pnl":"Mean P&L","std_pnl":"Std Dev","sharpe":"Sharpe",
        "var_95":"VaR 95%","cvar_95":"CVaR 95%","max_loss":"Max Loss",
        "pct_loss":"% Loss","avg_tc":"Avg TC",
    })
    .style
    .format({
        "Mean P&L": "{:+.4f}", "Std Dev": "{:.4f}",
        "Sharpe": "{:+.3f}", "VaR 95%": "{:+.4f}",
        "CVaR 95%": "{:+.4f}", "Max Loss": "{:+.4f}",
        "% Loss": "{:.1%}", "Avg TC": "{:.4f}",
    })
    .map(_color_pnl, subset=["Mean P&L", "VaR 95%", "CVaR 95%", "Max Loss"])
    .background_gradient(subset=["Sharpe"], cmap="RdYlGn", vmin=-1, vmax=1)
)
st.dataframe(styled, width="stretch", height=420)


# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sharpe Comparison",
    "Risk-Return Map",
    "VaR / CVaR",
    "Transaction Costs",
    "P&L Distribution",
])

with tab1:
    st.plotly_chart(sharpe_bars(df), width="stretch", config={"displayModeBar": False})
    st.caption("Higher Sharpe = better risk-adjusted return. RL agents should lead in non-standard scenarios.")

with tab2:
    st.plotly_chart(risk_return_scatter(df), width="stretch", config={"displayModeBar": False})
    st.caption("Top-left corner = best: high return, low risk. Each dot is one strategy × scenario combination.")

with tab3:
    sc_for_var = selected_scenario if selected_scenario != "All" else "base"
    st.plotly_chart(var_cvar_bars(df, sc_for_var), width="stretch", config={"displayModeBar": False})
    if selected_scenario == "All":
        st.caption("Showing Base scenario. Use the sidebar filter to view other scenarios.")

with tab4:
    st.plotly_chart(tc_comparison(df), width="stretch", config={"displayModeBar": False})
    st.caption("RL agents learn to avoid unnecessary rebalancing, especially under high transaction costs.")

with tab5:
    # Regenerate 200 episodes on-demand for the KDE
    if st.button("▶  Generate P&L Distribution (200 episodes)"):
        with st.spinner("Simulating 200 base-scenario episodes per strategy…"):
            env_kw = dict(s0=100, mu=0.05, sigma=0.20, dt=1/252, maturity=30/252,
                          strike=100, rate=0.01, transaction_cost=0.001)
            pnl_dict: dict[str, np.ndarray] = {}
            strategies_to_show = [s for s in ["no_hedge", "delta", "ppo", "sac"] if s in df["strategy"].values]
            models = {n: load_model(n) for n in ["ppo", "sac"]}
            for strat in strategies_to_show:
                pnls = []
                for ep in range(200):
                    ep_df = collect_episode(env_kw, strat, model=models.get(strat), seed=ep)
                    pnls.append(float(ep_df["portfolio_value"].iloc[-1]))
                pnl_dict[strat] = np.array(pnls)
        st.plotly_chart(pnl_distribution(pnl_dict), width="stretch", config={"displayModeBar": False})
    else:
        st.markdown('<div style="text-align:center;padding:40px;color:#4a5568;">Click the button to generate P&L distributions (takes ~15s).</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Key findings
# ─────────────────────────────────────────────

st.markdown('<div class="ws-section-header">KEY FINDINGS</div>', unsafe_allow_html=True)

_df_valid = df.dropna(subset=["sharpe", "var_95", "avg_tc"])
if _df_valid.empty:
    st.info("Insufficient data for key findings.")
    st.stop()
best_sharpe = _df_valid.loc[_df_valid["sharpe"].idxmax()]
best_var    = _df_valid.loc[_df_valid["var_95"].idxmax()]
least_tc    = _df_valid.loc[_df_valid["avg_tc"].idxmin()]

finding_cols = st.columns(3)
findings = [
    ("Best Risk-Adjusted Return",  f"{best_sharpe['strategy'].upper()} — {SCENARIO_LABEL.get(best_sharpe['scenario'], best_sharpe['scenario'])}", f"Sharpe: {best_sharpe['sharpe']:+.3f}"),
    ("Best Tail Protection",       f"{best_var['strategy'].upper()} — {SCENARIO_LABEL.get(best_var['scenario'], best_var['scenario'])}",          f"VaR 95%: {best_var['var_95']:+.4f}"),
    ("Lowest Transaction Cost",    f"{least_tc['strategy'].upper()} — {SCENARIO_LABEL.get(least_tc['scenario'], least_tc['scenario'])}",           f"Avg TC: {least_tc['avg_tc']:.4f}"),
]
for col, (title, name, stat) in zip(finding_cols, findings):
    with col:
        st.markdown(
            f'<div class="ws-card">'
            f'<div class="ws-card-title" style="color:#00d4aa;">{title}</div>'
            f'<div style="font-size:13px;color:#d4d8e2;margin:8px 0 4px;">{name}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#f5c518;">{stat}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
