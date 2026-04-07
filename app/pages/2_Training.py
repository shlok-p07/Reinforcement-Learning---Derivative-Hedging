"""Training Monitor — live progress bars, learning curves, model status."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import subprocess

from components.charts import inject_css, learning_curves, GREEN, GOLD  # noqa

st.set_page_config(page_title="Training | RL Hedging", page_icon=None, layout="wide")
st.markdown(inject_css(), unsafe_allow_html=True)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

TOTAL_STEPS = {"ppo": 500_000, "sac": 300_000}
AGENT_COLOR = {"ppo": GREEN, "sac": GOLD}
AGENT_LABEL = {
    "ppo": "PPO — Proximal Policy Optimisation",
    "sac": "SAC — Soft Actor-Critic",
}


def _load_curve(agent: str) -> dict | None:
    path = os.path.join(ROOT, "results", "learning_curves", agent, "evaluations.npz")
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=False)
        if "timesteps" not in d or "results" not in d:
            return None
        ts  = d["timesteps"]
        res = d["results"]
        if len(ts) == 0 or len(res) == 0 or len(ts) != len(res):
            return None
        return {"timesteps": ts, "results": res}
    except Exception:
        return None


def _model_exists(name: str) -> bool:
    return os.path.exists(os.path.join(ROOT, "models", f"{name}_hedger.zip"))


# ─────────────────────────────────────────────
# Session state — track which agents are running
# ─────────────────────────────────────────────

for _a in ("ppo", "sac"):
    if f"training_{_a}" not in st.session_state:
        st.session_state[f"training_{_a}"] = False


# ─────────────────────────────────────────────
# Live progress fragment — polls every 3 s
# Defined at top level so Streamlit can manage it consistently.
# ─────────────────────────────────────────────

@st.fragment(run_every="3s")
def _live_progress() -> None:
    """Re-runs every 3 s; reads evaluations.npz to drive progress bars."""
    active = [a for a in ("ppo", "sac") if st.session_state.get(f"training_{a}")]
    if not active:
        return

    st.markdown('<div class="ws-section-header">TRAINING IN PROGRESS</div>',
                unsafe_allow_html=True)

    all_done = True
    for agent in active:
        total = TOTAL_STEPS[agent]
        color = AGENT_COLOR[agent]
        data  = _load_curve(agent)

        if data is None:
            # Subprocess just launched — .npz not written yet
            all_done = False
            st.markdown(
                f'<div class="ws-card" style="margin-bottom:12px;">'
                f'<div class="ws-card-title" style="color:{color};">'
                f'{AGENT_LABEL[agent]}</div>'
                f'<div style="font-size:12px;color:#6b7a8d;margin:10px 0 6px;">'
                f'Starting up — building environments and initialising the neural network…'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(0.0, text="Waiting for first checkpoint…")
            continue

        current = int(data["timesteps"][-1])
        pct     = min(current / total, 1.0)
        mean_r  = float(data["results"][-1].mean())
        best_r  = float(data["results"].mean(axis=1).max())
        done    = pct >= 1.0 and _model_exists(agent)

        if not done:
            all_done = False

        status_text  = "COMPLETE" if done else "TRAINING"
        status_color = GREEN if done else GOLD

        st.markdown(
            f'<div class="ws-card" style="margin-bottom:12px;">'
            f'<div class="ws-card-title" style="color:{color};">'
            f'{AGENT_LABEL[agent]} &nbsp;'
            f'<span style="color:{status_color};font-size:9px;'
            f'letter-spacing:1.5px;">{status_text}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        bar_label = (
            f"{current:,} / {total:,} steps  —  {pct:.0%} complete"
            if not done
            else f"Training finished — {current:,} steps completed"
        )
        st.progress(pct, text=bar_label)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Progress",        f"{pct:.1%}")
        c2.metric("Steps Completed", f"{current:,}")
        c3.metric("Latest Reward",   f"{mean_r:+.4f}")
        c4.metric("Best Reward",     f"{best_r:+.4f}")

        if done:
            st.success(
                f"{agent.upper()} training complete. "
                "The model is ready — head to Live Demo or Evaluation to use it."
            )

    st.caption("Progress updates every 3 seconds automatically.")

    # Clear session state AFTER rendering — avoids triggering a full rerun mid-fragment
    if all_done:
        for agent in active:
            st.session_state[f"training_{agent}"] = False
        st.rerun()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## TRAINING CONTROLS")
    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

    st.markdown("**Model Status**")
    for name in ["ppo", "sac"]:
        ok = _model_exists(name)
        badge = (
            f'<span class="badge-live">● TRAINED</span>'
            if ok else
            f'<span class="badge-off">○ NOT FOUND</span>'
        )
        running = st.session_state.get(f"training_{name}", False)
        if running:
            badge = '<span class="badge-warn">◐ TRAINING</span>'
        st.markdown(f"{name.upper()} &nbsp;&nbsp; {badge}", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)
    st.markdown("**Retrain Models**")
    st.warning("Training takes 30-60 min per model. Existing models will be overwritten.")

    train_ppo_btn  = st.button("Train PPO (500k steps)", width="stretch")
    train_sac_btn  = st.button("Train SAC (300k steps)", width="stretch")
    train_both_btn = st.button("Train Both", type="primary", width="stretch")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown('<div class="ws-title">TRAINING MONITOR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ws-subtitle">Live training progress · Reward curves · Model hyperparameters</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Launch training processes
# ─────────────────────────────────────────────

if train_ppo_btn or train_both_btn:
    script = os.path.join(ROOT, "training", "train_ppo.py")
    if not os.path.exists(script):
        st.error(f"Training script not found: {script}")
    else:
        try:
            # Remove stale checkpoint so fragment shows real progress, not old 100%
            old_npz = os.path.join(ROOT, "results", "learning_curves", "ppo", "evaluations.npz")
            if os.path.exists(old_npz):
                os.remove(old_npz)
            subprocess.Popen([sys.executable, script], cwd=ROOT)
            st.session_state.training_ppo = True
        except Exception as exc:
            st.error(f"Failed to launch PPO training: {exc}")

if train_sac_btn or train_both_btn:
    script = os.path.join(ROOT, "training", "train_sac.py")
    if not os.path.exists(script):
        st.error(f"Training script not found: {script}")
    else:
        try:
            # Remove stale checkpoint so fragment shows real progress, not old 100%
            old_npz = os.path.join(ROOT, "results", "learning_curves", "sac", "evaluations.npz")
            if os.path.exists(old_npz):
                os.remove(old_npz)
            subprocess.Popen([sys.executable, script], cwd=ROOT)
            st.session_state.training_sac = True
        except Exception as exc:
            st.error(f"Failed to launch SAC training: {exc}")


# ─────────────────────────────────────────────
# Live progress bars (auto-refresh via fragment)
# ─────────────────────────────────────────────

_live_progress()


# ─────────────────────────────────────────────
# Learning curves (static — reloaded on page visit)
# ─────────────────────────────────────────────

st.markdown('<div class="ws-section-header">LEARNING CURVES</div>', unsafe_allow_html=True)

ppo_data = _load_curve("ppo")
sac_data = _load_curve("sac")

if ppo_data is None and sac_data is None:
    st.markdown(
        """<div class="ws-card" style="text-align:center;padding:40px;">
          <div style="font-size:11px;letter-spacing:3px;color:#1e3a5f;
               font-family:'JetBrains Mono',monospace;">NO DATA</div>
          <div style="font-size:14px;color:#6b7a8d;margin-top:12px;">
            No training history yet. Start training above to see reward curves appear here.
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    fig = learning_curves(ppo_data, sac_data)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="ws-section-header">TRAINING STATISTICS</div>',
                unsafe_allow_html=True)
    stat_cols = st.columns(2)

    for col, (name, data, color) in zip(
        stat_cols, [("PPO", ppo_data, GREEN), ("SAC", sac_data, GOLD)]
    ):
        if data is None:
            continue
        final_mean = float(data["results"][-1].mean())
        final_std  = float(data["results"][-1].std())
        best_mean  = float(data["results"].mean(axis=1).max())
        n_evals    = len(data["timesteps"])
        total_ts   = int(data["timesteps"][-1])

        with col:
            st.markdown(
                f"""<div class="ws-card">
                  <div class="ws-card-title" style="color:{color};">{name} TRAINING STATS</div>
                  <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e3a5f;">
                    <span style="color:#6b7a8d;font-size:12px;">Final Mean Reward</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{color};">{final_mean:+.4f}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e3a5f;">
                    <span style="color:#6b7a8d;font-size:12px;">Final Std Dev</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#d4d8e2;">{final_std:.4f}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e3a5f;">
                    <span style="color:#6b7a8d;font-size:12px;">Best Mean Reward</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#00d4aa;">{best_mean:+.4f}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e3a5f;">
                    <span style="color:#6b7a8d;font-size:12px;">Steps Trained</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#d4d8e2;">{total_ts:,}</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;padding:6px 0;">
                    <span style="color:#6b7a8d;font-size:12px;">Evaluation Checkpoints</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#d4d8e2;">{n_evals}</span>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# Hyperparameter reference
# ─────────────────────────────────────────────

st.markdown('<div class="ws-section-header">HYPERPARAMETERS</div>', unsafe_allow_html=True)

hp_col1, hp_col2 = st.columns(2)

with hp_col1:
    st.markdown(
        """<div class="ws-card">
          <div class="ws-card-title" style="color:#00d4aa;">PPO — PROXIMAL POLICY OPTIMISATION</div>
          <table class="ws-table" style="margin-top:8px;">
            <tr><td style="color:#6b7a8d;">Learning Rate</td><td>1e-4</td></tr>
            <tr><td style="color:#6b7a8d;">n_steps</td><td>4,096</td></tr>
            <tr><td style="color:#6b7a8d;">Batch Size</td><td>256</td></tr>
            <tr><td style="color:#6b7a8d;">n_epochs</td><td>10</td></tr>
            <tr><td style="color:#6b7a8d;">gamma (discount)</td><td>0.99</td></tr>
            <tr><td style="color:#6b7a8d;">GAE lambda</td><td>0.95</td></tr>
            <tr><td style="color:#6b7a8d;">Clip range</td><td>0.20</td></tr>
            <tr><td style="color:#6b7a8d;">Entropy coef</td><td>0.005</td></tr>
            <tr><td style="color:#6b7a8d;">Parallel envs</td><td>8</td></tr>
            <tr><td style="color:#6b7a8d;">Network</td><td>[256, 256] MLP</td></tr>
            <tr><td style="color:#6b7a8d;">Total steps</td><td>500,000</td></tr>
          </table>
        </div>""",
        unsafe_allow_html=True,
    )

with hp_col2:
    st.markdown(
        """<div class="ws-card">
          <div class="ws-card-title" style="color:#f5c518;">SAC — SOFT ACTOR-CRITIC</div>
          <table class="ws-table" style="margin-top:8px;">
            <tr><td style="color:#6b7a8d;">Learning Rate</td><td>3e-4</td></tr>
            <tr><td style="color:#6b7a8d;">Buffer Size</td><td>200,000</td></tr>
            <tr><td style="color:#6b7a8d;">Batch Size</td><td>256</td></tr>
            <tr><td style="color:#6b7a8d;">Learning Starts</td><td>2,000</td></tr>
            <tr><td style="color:#6b7a8d;">gamma (discount)</td><td>0.99</td></tr>
            <tr><td style="color:#6b7a8d;">tau (soft update)</td><td>0.005</td></tr>
            <tr><td style="color:#6b7a8d;">Entropy coef</td><td>auto (adaptive)</td></tr>
            <tr><td style="color:#6b7a8d;">SDE exploration</td><td>Yes</td></tr>
            <tr><td style="color:#6b7a8d;">Network</td><td>[256, 256] MLP</td></tr>
            <tr><td style="color:#6b7a8d;">Total steps</td><td>300,000</td></tr>
          </table>
        </div>""",
        unsafe_allow_html=True,
    )
