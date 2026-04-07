"""Shared Plotly chart builders — Wall Street dark theme."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────
BG        = "#070d1a"
BG2       = "#0d1b2a"
BORDER    = "#1e3a5f"
GREEN     = "#00d4aa"
RED       = "#ff4b4b"
GOLD      = "#f5c518"
BLUE      = "#4a9eff"
PURPLE    = "#9b5de5"
GREY      = "#4a5568"
TEXT      = "#d4d8e2"
TEXT_DIM  = "#6b7a8d"

STRATEGY_COLOR = {
    "no_hedge": RED,
    "random":   GREY,
    "delta":    BLUE,
    "sac":      GOLD,
    "ppo":      GREEN,
    "delta_hedge": BLUE,
}

SCENARIO_LABEL = {
    "base":          "Base Market",
    "high_tc":       "High Transaction Cost",
    "vol_mismatch":  "Volatility Mismatch",
    "regime_switch": "Regime Switching",
}

_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG2,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(bgcolor=BG2, bordercolor=BORDER, borderwidth=1),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
)


def _apply(fig) -> go.Figure:
    fig.update_layout(**_LAYOUT)
    return fig


# ──────────────────────────────────────────────
# Live episode chart (4-panel)
# ──────────────────────────────────────────────

def live_episode_chart(df: pd.DataFrame, strategy: str = "agent") -> go.Figure:
    """4-panel live hedging dashboard built incrementally."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, height=520,
                          title_text="No episode data", title_font=dict(color=RED))
        return fig

    color = STRATEGY_COLOR.get(strategy, GREEN)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Asset Price", "Hedge Position vs Delta",
                        "Portfolio Value (P&L)", "Step Hedging Error"],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    steps = df["step"].tolist()

    # ── Price path ──
    fig.add_trace(go.Scatter(
        x=steps, y=df["spot"], name="Spot Price",
        line=dict(color=BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(74,158,255,0.07)",
    ), row=1, col=1)
    if "strike" in df.columns and len(df) > 0:
        fig.add_hline(y=float(df["strike"].iloc[0]), line_dash="dot",
                      line_color=GOLD, opacity=0.6, row=1, col=1)

    # ── Hedge vs Delta ──
    fig.add_trace(go.Scatter(
        x=steps, y=df["delta"], name="BS Delta",
        line=dict(color=BLUE, width=1.5, dash="dot"),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=steps, y=df["hedge_position"], name="RL Hedge",
        line=dict(color=color, width=2.5),
        fill="tonexty", fillcolor=f"rgba(0,212,170,0.08)",
    ), row=1, col=2)

    # ── Portfolio value — line color and fill reflect gain/loss sign ──
    pv = df["portfolio_value"].tolist()
    final_pv = pv[-1] if pv else 0.0
    pv_color = GREEN if final_pv >= 0 else RED
    pv_fill = "rgba(0,212,170,0.10)" if final_pv >= 0 else "rgba(255,75,75,0.10)"
    fig.add_trace(go.Scatter(
        x=steps, y=pv, name="Portfolio V",
        line=dict(color=pv_color, width=2.5),
        fill="tozeroy",
        fillcolor=pv_fill,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color=GREY, line_dash="dot", opacity=0.5, row=2, col=1)

    # ── Step hedging error ──
    if "delta_v" in df.columns:
        dv = df["delta_v"].fillna(0).tolist()
        bar_colors = [GREEN if v >= 0 else RED for v in dv]
        fig.add_trace(go.Bar(
            x=steps, y=dv, name="ΔV",
            marker_color=bar_colors, opacity=0.8,
        ), row=2, col=2)
        fig.add_hline(y=0, line_color=GREY, line_dash="dot", opacity=0.5, row=2, col=2)

    fig.update_layout(
        **_LAYOUT,
        height=520,
        showlegend=True,
        title_text=f"Live Hedging Episode — Strategy: <b>{strategy.upper()}</b>",
        title_font=dict(color=GREEN, size=14),
    )
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_DIM
        ann.font.size = 11
    return fig


# ──────────────────────────────────────────────
# P&L distribution
# ──────────────────────────────────────────────

def pnl_distribution(pnls_by_strategy: dict[str, np.ndarray]) -> go.Figure:
    """Overlapping KDE curves + VaR markers."""
    from scipy.stats import gaussian_kde  # local import

    fig = go.Figure()
    for strategy, pnls in pnls_by_strategy.items():
        pnls = np.asarray(pnls, dtype=float)
        pnls = pnls[np.isfinite(pnls)]       # drop NaN / inf
        if len(pnls) < 2:
            continue                           # need at least 2 points for KDE
        color = STRATEGY_COLOR.get(strategy, GREY)
        try:
            kde = gaussian_kde(pnls, bw_method=0.3)
        except np.linalg.LinAlgError:
            continue                           # singular matrix (all identical values)
        xs = np.linspace(pnls.min() - 1, pnls.max() + 1, 400)
        ys = kde(xs)
        var95 = float(np.percentile(pnls, 5))

        fig.add_trace(go.Scatter(
            x=xs, y=ys, name=strategy.upper(),
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
        ))
        fig.add_vline(
            x=var95, line_color=color, line_dash="dash", line_width=1.5,
            annotation_text=f"VaR {var95:.2f}",
            annotation_font_color=color, annotation_font_size=10,
        )

    fig.add_vline(x=0, line_color=GREY, line_dash="dot", opacity=0.5)
    fig.update_layout(
        **_LAYOUT, height=400,
        title="Terminal P&L Distribution  (dashed = 95% VaR)",
        title_font=dict(color=GREEN),
        xaxis_title="Terminal Portfolio P&L",
        yaxis_title="Density",
    )
    return fig


# ──────────────────────────────────────────────
# Sharpe bar chart
# ──────────────────────────────────────────────

def sharpe_bars(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of Sharpe ratios by scenario."""
    scenarios = list(SCENARIO_LABEL.keys())
    strategies = [s for s in STRATEGY_COLOR if s in df["strategy"].values]

    fig = go.Figure()
    for strategy in strategies:
        vals = []
        for sc in scenarios:
            row = df[(df["strategy"] == strategy) & (df["scenario"] == sc)]
            if len(row) and not pd.isna(row["sharpe"].values[0]):
                vals.append(float(row["sharpe"].values[0]))
            else:
                vals.append(0.0)
        color = STRATEGY_COLOR.get(strategy, GREY)
        fig.add_trace(go.Bar(
            name=strategy.upper(), x=[SCENARIO_LABEL[s] for s in scenarios], y=vals,
            marker_color=color, opacity=0.85,
            text=[f"{v:.2f}" for v in vals], textposition="outside",
            textfont=dict(size=10, color=TEXT),
        ))

    fig.add_hline(y=0, line_color=GREY, line_dash="dot", opacity=0.5)
    fig.update_layout(
        **_LAYOUT, height=420,
        title="Sharpe Ratio by Strategy and Scenario",
        title_font=dict(color=GREEN),
        yaxis_title="Sharpe  (mean / std P&L)",
        barmode="group",
        bargap=0.15, bargroupgap=0.05,
    )
    return fig


# ──────────────────────────────────────────────
# Risk-return scatter
# ──────────────────────────────────────────────

def risk_return_scatter(df: pd.DataFrame) -> go.Figure:
    """Mean P&L vs Std P&L — each dot is a strategy × scenario."""
    fig = go.Figure()
    scenario_symbols = {"base": "circle", "high_tc": "square",
                        "vol_mismatch": "diamond", "regime_switch": "triangle-up"}

    for _, row in df.iterrows():
        color = STRATEGY_COLOR.get(row["strategy"], GREY)
        symbol = scenario_symbols.get(row["scenario"], "circle")
        fig.add_trace(go.Scatter(
            x=[row["std_pnl"]], y=[row["mean_pnl"]],
            mode="markers+text",
            marker=dict(color=color, symbol=symbol, size=14,
                        line=dict(color=BG2, width=1.5)),
            text=[row["strategy"].upper()],
            textposition="top right",
            textfont=dict(color=color, size=9),
            name=f"{row['strategy']} / {row['scenario']}",
            hovertemplate=(
                f"<b>{row['strategy'].upper()}</b> — {SCENARIO_LABEL.get(row['scenario'], row['scenario'])}<br>"
                f"Mean P&L: {row['mean_pnl']:+.4f}<br>"
                f"Std P&L:  {row['std_pnl']:.4f}<br>"
                f"Sharpe:   {row['sharpe']:+.3f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.add_hline(y=0, line_color=GREY, line_dash="dot", opacity=0.4)
    fig.update_layout(
        **_LAYOUT, height=460,
        title="Risk-Return Map  (top-left = optimal: high return, low risk)",
        title_font=dict(color=GREEN),
        xaxis_title="P&L Std Dev  (risk increases →)",
        yaxis_title="Mean Terminal P&L  (return increases ↑)",
    )
    return fig


# ──────────────────────────────────────────────
# VaR / CVaR comparison
# ──────────────────────────────────────────────

def var_cvar_bars(df: pd.DataFrame, scenario: str = "base") -> go.Figure:
    """Side-by-side VaR-95 and CVaR-95 shown as positive loss magnitude (standard risk convention)."""
    sub = df[df["scenario"] == scenario]
    strategies = [s for s in STRATEGY_COLOR if s in sub["strategy"].values]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["95% VaR — Loss at 5th Percentile", "95% CVaR — Expected Shortfall"],
    )

    for metric, col in [("var_95", 1), ("cvar_95", 2)]:
        raw = []
        for s in strategies:
            filtered = sub[sub["strategy"] == s]
            if len(filtered) and not pd.isna(filtered[metric].values[0]):
                raw.append(float(filtered[metric].values[0]))
            else:
                raw.append(0.0)
        # Negate: VaR/CVaR stored as negative P&L; display as positive loss magnitude
        loss = [-v for v in raw]
        bar_colors = [STRATEGY_COLOR.get(s, GREY) for s in strategies]
        fig.add_trace(go.Bar(
            x=[s.upper() for s in strategies],
            y=loss,
            marker_color=bar_colors,
            opacity=0.85,
            text=[f"{v:.4f}" for v in loss],
            textposition="outside",
            textfont=dict(size=10, color=TEXT),
            showlegend=False,
        ), row=1, col=col)

    fig.update_layout(
        **_LAYOUT, height=400,
        title=f"Tail Loss Magnitude — {SCENARIO_LABEL.get(scenario, scenario)}  (taller bar = greater downside risk)",
        title_font=dict(color=GREEN),
    )
    # Set y-axis titles via update_yaxes to avoid conflict with _LAYOUT's yaxis key
    fig.update_yaxes(title_text="Loss Magnitude")
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_DIM
    return fig


# ──────────────────────────────────────────────
# Learning curves
# ──────────────────────────────────────────────

def learning_curves(ppo_data: dict | None, sac_data: dict | None) -> go.Figure:
    """Training reward curves with confidence bands."""
    fig = go.Figure()

    for name, data, color in [("PPO", ppo_data, GREEN), ("SAC", sac_data, GOLD)]:
        if data is None:
            continue
        ts    = data["timesteps"]
        means = data["results"].mean(axis=1)
        stds  = data["results"].std(axis=1)
        fig.add_trace(go.Scatter(
            x=ts, y=means, name=name,
            line=dict(color=color, width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([ts, ts[::-1]]),
            y=np.concatenate([means + stds, (means - stds)[::-1]]),
            fill="toself", fillcolor=f"rgba({_hex_to_rgb(color)},0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        **_LAYOUT, height=380,
        title="Training Reward Curves  (mean ± 1 std over eval episodes)",
        title_font=dict(color=GREEN),
        xaxis_title="Training Timesteps",
        yaxis_title="Mean Episode Reward",
    )
    return fig


# ──────────────────────────────────────────────
# Transaction cost comparison
# ──────────────────────────────────────────────

def tc_comparison(df: pd.DataFrame) -> go.Figure:
    """Average TC paid per episode across scenarios."""
    scenarios = list(SCENARIO_LABEL.keys())
    strategies = [s for s in STRATEGY_COLOR if s in df["strategy"].values]

    fig = go.Figure()
    for strategy in strategies:
        vals = []
        for sc in scenarios:
            row = df[(df["strategy"] == strategy) & (df["scenario"] == sc)]
            vals.append(float(row["avg_tc"].values[0]) if len(row) else 0.0)
        fig.add_trace(go.Bar(
            name=strategy.upper(),
            x=[SCENARIO_LABEL[s] for s in scenarios], y=vals,
            marker_color=STRATEGY_COLOR.get(strategy, GREY), opacity=0.85,
        ))

    fig.update_layout(
        **_LAYOUT, height=360,
        title="Average Transaction Costs per Episode  (RL learns to rebalance less)",
        title_font=dict(color=GREEN),
        yaxis_title="Avg Total TC",
        barmode="group",
    )
    return fig


# ──────────────────────────────────────────────
# CSS injection
# ──────────────────────────────────────────────

WS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Metric cards ── */
.ws-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 6px 0;
}
.ws-card-title {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6b7a8d;
    margin-bottom: 6px;
}
.ws-card-value {
    font-size: 26px;
    font-weight: 600;
    color: #d4d8e2;
    font-family: 'JetBrains Mono', monospace;
}
.ws-card-sub {
    font-size: 11px;
    color: #6b7a8d;
    margin-top: 4px;
}
.positive { color: #00d4aa !important; }
.negative { color: #ff4b4b !important; }
.neutral  { color: #f5c518 !important; }

/* ── Section headers ── */
.ws-section-header {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00d4aa;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 24px 0 14px;
}

/* ── Status badges ── */
.badge-live   { background:#003d2e; color:#00d4aa; padding:2px 10px; border-radius:12px; font-size:10px; border:1px solid #00d4aa; }
.badge-warn   { background:#3d2e00; color:#f5c518; padding:2px 10px; border-radius:12px; font-size:10px; border:1px solid #f5c518; }
.badge-off    { background:#2a0d0d; color:#ff4b4b; padding:2px 10px; border-radius:12px; font-size:10px; border:1px solid #ff4b4b; }

/* ── Ticker strip ── */
.ticker { font-family:'JetBrains Mono',monospace; font-size:12px; color:#d4d8e2; }

/* ── Table ── */
.ws-table { border-collapse:collapse; width:100%; }
.ws-table th { background:#0d1b2a; color:#6b7a8d; font-size:10px; letter-spacing:1px; text-transform:uppercase; padding:8px 12px; border-bottom:1px solid #1e3a5f; }
.ws-table td { padding:9px 12px; border-bottom:1px solid #0d1b2a; font-family:'JetBrains Mono',monospace; font-size:12px; }
.ws-table tr:hover td { background:#0d1b2a; }

/* ── Divider ── */
.ws-divider { border:none; border-top:1px solid #1e3a5f; margin:20px 0; }

/* ── Main title ── */
.ws-title { font-size:28px; font-weight:600; letter-spacing:1px; color:#d4d8e2; }
.ws-subtitle { font-size:13px; color:#6b7a8d; letter-spacing:0.5px; margin-top:4px; }
</style>
"""


def inject_css() -> str:
    return WS_CSS


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def fmt_pnl(v: float) -> str:
    cls = "positive" if v >= 0 else "negative"
    sign = "+" if v >= 0 else ""
    return f'<span class="{cls}">{sign}{v:.4f}</span>'


def fmt_sharpe(v: float) -> str:
    cls = "positive" if v > 0.1 else ("negative" if v < -0.1 else "neutral")
    return f'<span class="{cls}">{v:+.3f}</span>'
