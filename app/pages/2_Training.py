"""Training Monitor — live progress bars, learning curves, model status."""

import json
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import streamlit as st
import subprocess

from components.charts import inject_css, learning_curves, GREEN, GOLD  # noqa

st.set_page_config(page_title="Training | RL Hedging", page_icon=None, layout="wide")
st.markdown(inject_css(), unsafe_allow_html=True)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

TOTAL_STEPS = {"ppo": 500_000, "sac": 300_000}
AGENT_COLOR = {"ppo": GREEN, "sac": GOLD}
AGENT_LABEL = {
    "ppo": "PPO — Proximal Policy Optimization",
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


def _load_progress(agent: str) -> dict | None:
    """Read the lightweight progress.json written every ~2k steps by ProgressFileCallback."""
    path = os.path.join(ROOT, "results", "learning_curves", agent, "progress.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if "timesteps" not in data:
            return None
        return data
    except Exception:
        return None


def _model_exists(name: str) -> bool:
    return os.path.exists(os.path.join(ROOT, "models", f"{name}_hedger.zip"))


def _pid_path(agent: str) -> str:
    return os.path.join(ROOT, "results", "learning_curves", agent, "training.pid")


def _pid_alive(agent: str) -> bool:
    """Return True only if the training PID file exists AND the process is still running."""
    pid_file = _pid_path(agent)
    if not os.path.exists(pid_file):
        return False
    try:
        with open(pid_file, encoding="utf-8") as fh:
            pid = int(fh.read().strip())
        os.kill(pid, 0)  # signal 0 = existence check only, raises OSError if dead
        return True
    except (OSError, ValueError, ProcessLookupError):
        return False


def _kill_existing(agent: str) -> None:
    """Kill a still-running training subprocess for this agent (if any).

    Without this, re-clicking 'Retrain' while a run is in progress leaves the old
    process alive.  Both processes then write to the same progress.json, causing
    the bar to bounce backwards as each overwrites the other's values.
    """
    pid_file = _pid_path(agent)
    if not os.path.exists(pid_file):
        return
    try:
        with open(pid_file, encoding="utf-8") as fh:
            pid = int(fh.read().strip())
        os.kill(pid, signal.SIGKILL)  # SIGKILL: immediate kernel termination, cannot be delayed or blocked
    except (OSError, ValueError, ProcessLookupError):
        pass  # already dead or unreadable
    finally:
        try:
            os.remove(pid_file)
        except OSError:
            pass


def _stale_files(agent: str) -> list[str]:
    """Return paths of checkpoint/progress files to wipe before a fresh training run."""
    base = os.path.join(ROOT, "results", "learning_curves", agent)
    return [
        os.path.join(base, "evaluations.npz"),
        os.path.join(base, "progress.json"),
    ]


# Session state — track which agents are running

for _a in ("ppo", "sac"):
    if f"training_{_a}" not in st.session_state:
        st.session_state[f"training_{_a}"] = False


@st.fragment(run_every="2s")
def _live_progress() -> None:
    """Re-runs every 2 s; reads progress.json (fast) and evaluations.npz (rewards)."""
    # Auto-correct stale session flags: if the subprocess is no longer running
    # (e.g. training finished, page was reloaded, or process was killed), clear
    # the flag so the progress bar disappears and the completed stats show instead.
    for _a in ("ppo", "sac"):
        if st.session_state.get(f"training_{_a}") and not _pid_alive(_a):
            st.session_state[f"training_{_a}"] = False

    active = [a for a in ("ppo", "sac") if st.session_state.get(f"training_{a}")]
    if not active:
        return

    st.markdown('<div class="ws-section-header">TRAINING IN PROGRESS</div>',
                unsafe_allow_html=True)

    all_done = True
    for agent in active:
        total = TOTAL_STEPS[agent]
        color = AGENT_COLOR[agent]
        prog  = _load_progress(agent)   # lightweight JSON — updates every ~2k steps
        data  = _load_curve(agent)      # NPZ — updates every ~10k steps (has rewards)

        # Neither file exists yet — subprocess still initializing
        if prog is None and data is None:
            all_done = False
            st.markdown(
                f'<div class="ws-card" style="margin-bottom:12px;">'
                f'<div class="ws-card-title" style="color:{color};">'
                f'{AGENT_LABEL[agent]}</div>'
                f'<div style="font-size:12px;color:#6b7a8d;margin:10px 0 6px;">'
                f'Starting up — building environments and initializing the neural network…'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(0.0, text="Waiting for first steps…")
            continue

        # Use progress.json for step count (more frequent), NPZ for rewards (less frequent)
        if prog is not None:
            current = int(prog["timesteps"])
        elif data is not None:
            current = int(data["timesteps"][-1])
        else:
            current = 0

        pct  = min(current / total, 1.0)
        done = pct >= 1.0 and _model_exists(agent)

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

        phase = (prog or {}).get("phase", "collecting")
        phase_tag = {
            "updating":  " · Running policy updates…",
            "starting":  " · Initializing…",
            "collecting": "",
        }.get(phase, "")
        bar_label = (
            f"{current:,} / {total:,} steps  —  {pct:.1%}{phase_tag}"
            if not done
            else f"Training finished — {current:,} steps completed"
        )
        st.progress(pct, text=bar_label)

        # Reward metrics — only available once NPZ has at least one checkpoint
        if data is not None:
            mean_r = float(data["results"][-1].mean())
            best_r = float(data["results"].mean(axis=1).max())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Progress",        f"{pct:.1%}")
            c2.metric("Steps Completed", f"{current:,}")
            c3.metric("Latest Reward",   f"{mean_r:+.4f}")
            c4.metric("Best Reward",     f"{best_r:+.4f}")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Progress",        f"{pct:.1%}")
            c2.metric("Steps Completed", f"{current:,}")

        if done:
            st.success(
                f"{agent.upper()} training complete. "
                "The model is ready — head to Live Demo or Evaluation to use it."
            )

    st.caption("Progress updates every 2 seconds automatically.")

    # Clear session state AFTER rendering — avoids triggering a full rerun mid-fragment
    if all_done:
        for agent in active:
            st.session_state[f"training_{agent}"] = False
        st.rerun()


# Sidebar

with st.sidebar:
    st.markdown("## TRAINING CONTROLS")
    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

    st.markdown("**Model Status**")
    for name in ["ppo", "sac"]:
        ok = _model_exists(name)
        badge = (
            '<span class="badge-live">● TRAINED</span>'
            if ok else
            '<span class="badge-off">○ NOT FOUND</span>'
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


# Header

st.markdown('<div class="ws-title">TRAINING MONITOR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ws-subtitle">Live training progress · Reward curves · Model hyperparameters</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)


# Launch training processes

def _launch(agent: str) -> bool:
    """Start a training subprocess, returning True on success."""
    _kill_existing(agent)  # stop any previous run to prevent two processes racing on progress.json
    script = os.path.join(ROOT, "training", f"train_{agent}.py")
    if not os.path.exists(script):
        st.error(f"Training script not found: {script}")
        return False
    try:
        for stale in _stale_files(agent):
            if os.path.exists(stale):
                os.remove(stale)
        # Redirect stdout/stderr to DEVNULL — SB3 + tqdm output would fill the OS
        # pipe buffer and block the subprocess once the parent (Streamlit) stops
        # draining it.  Progress is tracked via progress.json instead.
        proc = subprocess.Popen(
            [sys.executable, script],
            cwd=os.path.abspath(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Persist PID so a future retrain click can terminate this process
        os.makedirs(os.path.dirname(_pid_path(agent)), exist_ok=True)
        with open(_pid_path(agent), "w", encoding="utf-8") as fh:
            fh.write(str(proc.pid))
        return True
    except Exception as exc:
        st.error(f"Failed to launch {agent.upper()} training: {exc}")
        return False


# Guard: don't re-launch an agent that is already running
if (train_ppo_btn or train_both_btn) and not st.session_state.get("training_ppo"):
    if _launch("ppo"):
        st.session_state.training_ppo = True

if (train_sac_btn or train_both_btn) and not st.session_state.get("training_sac"):
    if _launch("sac"):
        st.session_state.training_sac = True


# Live progress bars (auto-refresh via fragment)

_live_progress()


@st.fragment(run_every="10s")
def _learning_curves_live() -> None:
    """Auto-refreshes every 10 s so curves appear live during training and instantly on completion."""
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
        return

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


_learning_curves_live()


# Hyperparameter reference

st.markdown('<div class="ws-section-header">HYPERPARAMETERS</div>', unsafe_allow_html=True)

hp_col1, hp_col2 = st.columns(2)

with hp_col1:
    st.markdown(
        """<div class="ws-card">
          <div class="ws-card-title" style="color:#00d4aa;">PPO — PROXIMAL POLICY OPTIMIZATION</div>
          <table class="ws-table" style="margin-top:8px;">
            <tr><td style="color:#6b7a8d;">Learning Rate</td><td>1e-4</td></tr>
            <tr><td style="color:#6b7a8d;">n_steps</td><td>2,048</td></tr>
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
