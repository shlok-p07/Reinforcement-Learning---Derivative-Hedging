"""Real Market Data — SPY price history, options chain, vol surface.
Live price strip auto-refreshes every second via st.fragment(run_every).
Data files auto-update on page load when stale (no manual refresh needed).
"""

import os, sys, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from components.charts import inject_css, _LAYOUT, GREEN, RED, GOLD, BLUE, GREY, BG, BG2, TEXT, TEXT_DIM  # noqa

st.set_page_config(page_title="Market Data | RL Hedging", page_icon=None, layout="wide")
st.markdown(inject_css(), unsafe_allow_html=True)

ROOT     = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_DIR = os.path.join(ROOT, "data")


# ── Auto-update on stale data ─────────────────────────────────────────────────

def _is_data_stale() -> bool:
    """True if calibrated_params.json is from a previous calendar day."""
    p = os.path.join(DATA_DIR, "calibrated_params.json")
    if not os.path.exists(p):
        return True
    try:
        with open(p) as f:
            d = json.load(f)
        return d.get("as_of", "") != datetime.date.today().isoformat()
    except Exception:
        return True


if "market_data_refreshed" not in st.session_state:
    st.session_state.market_data_refreshed = False

if not st.session_state.market_data_refreshed and _is_data_stale():
    with st.spinner("Auto-updating market data from Yahoo Finance…"):
        script = os.path.join(ROOT, "data", "generate_data.py")
        r = subprocess.run(
            [sys.executable, script], cwd=ROOT, capture_output=True, text=True
        )
        if r.returncode == 0:
            st.cache_data.clear()
        # Mark refreshed regardless — don't loop on a persistent error
        st.session_state.market_data_refreshed = True
    st.rerun()


# ── Cached loaders — CSV data refreshes every 5 min ──────────────────────────

@st.cache_data(ttl=300)
def _load(filename: str) -> pd.DataFrame | None:
    p = os.path.join(DATA_DIR, filename)
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(ttl=300)
def _load_params() -> dict | None:
    p = os.path.join(DATA_DIR, "calibrated_params.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=1)   # fetch fresh quote every second
def _fetch_live_price() -> dict:
    """Pull latest SPY quote from Yahoo Finance fast_info.

    fast_info.get() uses camelCase keys; attribute access uses snake_case.
    Use attribute access to avoid silent None returns.
    """
    fi = yf.Ticker("SPY").fast_info
    price      = float(getattr(fi, "last_price", None) or 0)
    prev_close = float(getattr(fi, "previous_close", None) or price)
    day_high   = float(getattr(fi, "day_high", None) or price)
    day_low    = float(getattr(fi, "day_low", None) or price)
    volume     = int(getattr(fi, "last_volume", None) or
                     getattr(fi, "three_month_average_volume", None) or 0)
    return dict(price=price, prev_close=prev_close,
                day_high=day_high, day_low=day_low, volume=volume)


# Sidebar

with st.sidebar:
    st.markdown("## MARKET DATA")
    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

    params = _load_params()
    if params:
        st.markdown("**Calibrated GBM Params**")
        st.markdown(f"""
| Param | Value |
|---|---|
| Ticker | **{params.get('ticker','SPY')}** |
| As of | {params.get('as_of','—')} |
| σ annual | **{params.get('sigma_annual',0):.1%}** |
| μ annual | **{params.get('mu_annual',0):.1%}** |
| σ 21d RVol | {params.get('rvol_21d_last',0):.1%} |
| Last Close | **${params.get('s0_last',0):.2f}** |
""")
    else:
        st.warning("No calibrated params. Run `python data/generate_data.py`")

    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)
    if st.button("Refresh All Data", width="stretch"):
        with st.spinner("Fetching from Yahoo Finance…"):
            script = os.path.join(ROOT, "data", "generate_data.py")
            import subprocess
            r = subprocess.run([sys.executable, script], cwd=ROOT,
                               capture_output=True, text=True)
            if r.returncode == 0:
                st.cache_data.clear()
                st.success("Data refreshed.")
            else:
                st.error(f"Error: {r.stderr[-300:]}")
        st.rerun()


# Header

ticker = params.get("ticker", "SPY") if params else "SPY"

st.markdown(f'<div class="ws-title">LIVE MARKET DATA — {ticker}</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="ws-subtitle">'
    'Real options chain · Implied vol surface · Price history · '
    '<span style="color:#00d4aa;">&#9679; Live price feed — continuous 1s update</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)


# Live metrics strip — 1s auto-refresh fragment

def _metric_card(title: str, val: str, cls: str) -> str:
    """Return one metric card as an HTML string (no individual st.markdown call)."""
    return (
        f'<div class="ws-card" style="padding:12px 16px;">'
        f'<div class="ws-card-title">{title}</div>'
        f'<div class="ws-card-value {cls}" style="font-size:18px;">{val}</div>'
        f'</div>'
    )


def _cards_row(metrics: list[tuple[str, str, str]]) -> str:
    """Wrap cards in a single-row CSS grid — one DOM node, zero per-column flicker."""
    inner = "".join(_metric_card(t, v, c) for t, v, c in metrics)
    return (
        f'<div style="display:grid;grid-template-columns:repeat({len(metrics)},1fr);'
        f'gap:10px;margin-bottom:4px;">{inner}</div>'
    )


@st.fragment(run_every="1s")
def _live_metrics_strip() -> None:
    now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:12]  # HH:MM:SS.mmm
    try:
        q       = _fetch_live_price()
        price   = q["price"]
        prev    = q["prev_close"]
        chg     = price - prev
        chg_pct = chg / prev if prev else 0.0
        cls     = "positive" if chg >= 0 else "negative"
        arrow   = "▲" if chg >= 0 else "▼"

        metrics = [
            ("LIVE PRICE", f"${price:.2f}",                          cls),
            ("DAY CHANGE", f"{arrow} {chg:+.2f}  ({chg_pct:+.2%})", cls),
            ("DAY HIGH",   f"${q['day_high']:.2f}",                  "neutral"),
            ("DAY LOW",    f"${q['day_low']:.2f}",                   "neutral"),
            ("VOLUME",     f"{q['volume']/1e6:.1f}M",                "neutral"),
            ("UPDATED",    now,                                       "neutral"),
        ]
        # Single st.markdown call = one DOM update, no per-column flicker
        st.markdown(_cards_row(metrics), unsafe_allow_html=True)

    except Exception:
        # Fallback to last CSV close when market is closed / no feed
        spy_df = _load("spy_daily.csv")
        if spy_df is not None and len(spy_df) >= 2:
            last_row = spy_df.iloc[-1]
            prev_row = spy_df.iloc[-2]
            chg      = last_row["close"] - prev_row["close"]
            chg_pct  = chg / prev_row["close"]
            cls      = "positive" if chg >= 0 else "negative"
            arrow    = "▲" if chg >= 0 else "▼"
            fb = [
                ("LAST CLOSE", f"${last_row['close']:.2f}",             cls),
                ("DAY CHANGE", f"{arrow} {chg:+.2f}  ({chg_pct:+.2%})", cls),
                ("21D RVOL",   f"{last_row['rvol_21d']:.1%}",            "neutral"),
                ("63D RVOL",   f"{last_row['rvol_63d']:.1%}",            "neutral"),
                ("MODEL σ",    f"{params.get('sigma_annual',0):.1%}" if params else "—", "neutral"),
                ("UPDATED",    now,                                       "neutral"),
            ]
            st.markdown(_cards_row(fb), unsafe_allow_html=True)
        else:
            st.warning("Live feed and CSV data unavailable.")


_live_metrics_strip()
st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)


# Static charts — loaded once from CSV files

spy        = _load("spy_daily.csv")
chain      = _load("option_chain_calls.csv")
returns_df = _load("spy_returns.csv")
vol_surface = _load("vol_surface.csv")

if spy is None or spy.empty:
    st.error("No market data found. Run: `python data/generate_data.py`")
    st.stop()

_required = {"date", "close", "rvol_21d", "rvol_63d", "volume"}
_missing = _required - set(spy.columns)
if _missing:
    st.error(f"SPY data file is missing columns: {sorted(_missing)}. Re-run `python data/generate_data.py`")
    st.stop()

if len(spy) < 2:
    st.error("SPY data has fewer than 2 rows — cannot compute day change. Refresh data.")
    st.stop()

last = spy.iloc[-1]

tab1, tab2, tab3, tab4 = st.tabs(["Price & Vol", "Options Chain", "Vol Surface", "Regime History"])


with tab1:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=["SPY Close Price",
                                        "Realized Volatility (annualized)",
                                        "Daily Volume"],
                        vertical_spacing=0.06)

    fig.add_trace(go.Scatter(
        x=spy["date"], y=spy["close"], name="Close",
        line=dict(color=BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(74,158,255,0.07)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=spy["date"], y=spy["rvol_21d"], name="21d RVol",
        line=dict(color=GREEN, width=1.8),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=spy["date"], y=spy["rvol_63d"], name="63d RVol",
        line=dict(color=GOLD, width=1.5, dash="dot"),
    ), row=2, col=1)
    if params:
        fig.add_hline(y=params["sigma_annual"], line_color=RED, line_dash="dash",
                      annotation_text=f"Model σ={params['sigma_annual']:.1%}",
                      row=2, col=1)

    fig.add_trace(go.Bar(
        x=spy["date"], y=spy["volume"] / 1e6, name="Volume (M)",
        marker_color=GREY, opacity=0.6,
    ), row=3, col=1)

    fig.update_layout(**_LAYOUT, height=580, showlegend=True,
                      title="SPY Price History & Realized Volatility (1 Year)",
                      title_font=dict(color=GREEN))
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_DIM
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


with tab2:
    if chain is None or chain.empty:
        st.info("No options data. Market may be closed or data unavailable.")
    else:
        spot_approx = float(last["close"])

        col_filt1, col_filt2 = st.columns(2)
        with col_filt1:
            all_expiries = sorted(chain["expiry"].unique())
            selected_exp = st.selectbox("Expiry", all_expiries)
        with col_filt2:
            iv_max = st.slider("Max Implied Vol filter", 0.1, 2.0, 1.5, step=0.1)

        chain_sub = chain[
            (chain["expiry"] == selected_exp) & (chain["impl_vol"] <= iv_max)
        ].copy()

        fig2 = make_subplots(rows=1, cols=2,
                             subplot_titles=["Implied Volatility Smile",
                                             "Open Interest by Strike"])

        fig2.add_trace(go.Scatter(
            x=chain_sub["strike"], y=chain_sub["impl_vol"],
            mode="lines+markers", name="IV",
            line=dict(color=GREEN, width=2),
            marker=dict(size=5, color=GREEN),
        ), row=1, col=1)
        fig2.add_vline(x=spot_approx, line_color=GOLD, line_dash="dash",
                       annotation_text=f"Spot ${spot_approx:.0f}", row=1, col=1)

        oi = chain_sub.dropna(subset=["open_interest"])
        oi_colors = [GREEN if k >= spot_approx else RED for k in oi["strike"]]
        fig2.add_trace(go.Bar(
            x=oi["strike"], y=oi["open_interest"] / 1000,
            name="OI (k)", marker_color=oi_colors, opacity=0.75,
        ), row=1, col=2)
        fig2.add_vline(x=spot_approx, line_color=GOLD, line_dash="dash", row=1, col=2)

        fig2.update_layout(**_LAYOUT, height=420,
                           title=f"SPY Options — Expiry {selected_exp}",
                           title_font=dict(color=GREEN))
        for ann in fig2.layout.annotations:
            ann.font.color = TEXT_DIM
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

        with st.expander("Full Chain Table", expanded=False):
            show_cols = ["strike","last_price","bid","ask","impl_vol",
                         "delta","gamma","open_interest","volume"]
            disp = chain_sub[[c for c in show_cols if c in chain_sub.columns]].copy()
            disp = disp.sort_values("strike").reset_index(drop=True)
            def _hl(row):
                if abs(row["strike"] - spot_approx) <= spot_approx * 0.01:
                    return ["background-color: #0d2a1a"] * len(row)
                return [""] * len(row)
            st.dataframe(disp.style.apply(_hl, axis=1), width="stretch", height=350)


with tab3:
    if vol_surface is None or vol_surface.empty:
        st.info("No vol surface data available.")
    else:
        pvt = vol_surface.pivot_table(
            index="strike", columns="days_to_exp", values="impl_vol", aggfunc="mean"
        ).dropna(how="all")

        if not pvt.empty:
            fig3 = go.Figure(data=go.Surface(
                z=pvt.values,
                x=pvt.columns.tolist(),
                y=pvt.index.tolist(),
                colorscale=[
                    [0.0, "#0d2a1a"], [0.3, "#00d4aa"],
                    [0.6, "#f5c518"], [1.0, "#ff4b4b"],
                ],
                showscale=True,
                colorbar=dict(title="IV", tickformat=".0%"),
            ))
            fig3.update_layout(
                **_LAYOUT, height=560,
                title="SPY Implied Volatility Surface",
                title_font=dict(color=GREEN),
                scene=dict(
                    xaxis_title="Days to Expiry",
                    yaxis_title="Strike",
                    zaxis_title="Implied Vol",
                    xaxis=dict(backgroundcolor=BG2),
                    yaxis=dict(backgroundcolor=BG2),
                    zaxis=dict(backgroundcolor=BG2),
                    bgcolor=BG,
                ),
            )
            st.plotly_chart(fig3, width="stretch")
        else:
            st.info("Not enough data to render the surface.")


with tab4:
    if returns_df is None or returns_df.empty:
        st.info("No returns data.")
    else:
        fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["SPY Close with Vol Regime",
                                             "21-Day Realized Vol"],
                             vertical_spacing=0.08)

        df_spy_merged = spy.set_index("date").join(
            returns_df.set_index("date")[["regime", "regime_label"]], how="left"
        ).reset_index()

        high = df_spy_merged[df_spy_merged["regime"] == 1]

        fig4.add_trace(go.Scatter(
            x=df_spy_merged["date"], y=df_spy_merged["close"],
            line=dict(color=BLUE, width=2), name="Price", showlegend=True,
        ), row=1, col=1)
        fig4.add_trace(go.Scatter(
            x=high["date"], y=high["close"],
            mode="markers", marker=dict(color=RED, size=3, opacity=0.4),
            name="High Vol Regime",
        ), row=1, col=1)
        fig4.add_trace(go.Scatter(
            x=returns_df["date"], y=returns_df["rvol_21d"],
            line=dict(color=GREEN, width=2), name="21d RVol",
        ), row=2, col=1)

        threshold = float(returns_df["rvol_21d"].median())
        fig4.add_hline(y=threshold, line_color=GOLD, line_dash="dash",
                       annotation_text=f"Regime threshold {threshold:.1%}",
                       row=2, col=1)

        fig4.update_layout(**_LAYOUT, height=500,
                           title="Vol Regime History  (red dots = high-vol periods)",
                           title_font=dict(color=GREEN), showlegend=True)
        for ann in fig4.layout.annotations:
            ann.font.color = TEXT_DIM
        st.plotly_chart(fig4, width="stretch", config={"displayModeBar": False})

        n_high = int((returns_df["regime"] == 1).sum())
        n_low  = int((returns_df["regime"] == 0).sum())
        total  = n_high + n_low
        if total > 0:
            low_pct  = f"{n_low  / total:.0%}"
            high_pct = f"{n_high / total:.0%}"
        else:
            low_pct = high_pct = "—"
        st.markdown(
            f'<div class="ws-card" style="margin-top:8px;">'
            f'<div class="ws-card-title" style="color:#00d4aa;">REGIME STATISTICS</div>'
            f'<div style="font-size:12px;color:#d4d8e2;display:flex;gap:40px;margin-top:8px;">'
            f'<span>Low Vol days: <b style="color:#00d4aa;">{n_low}</b> ({low_pct})</span>'
            f'<span>High Vol days: <b style="color:#ff4b4b;">{n_high}</b> ({high_pct})</span>'
            f'<span>Regime threshold: <b style="color:#f5c518;">{threshold:.1%}</b> annualized RVol</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
