"""
PPO training on real SPY market data (~1,200 distinct 30-day windows, 5-year history).

Loads and continues from models/ppo_hedger.zip if it exists; delete to retrain from scratch.
Hyperparams: n_steps=2048, batch_size=256, n_epochs=10, γ=0.99, λ=0.95, clip=0.2,
             ent_coef=0.005, lr=1e-4 (reduced to 5e-5 when fine-tuning).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.real_data_env import RealDataHedgingEnv

# Config

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "spy_daily.csv")

ENV_KWARGS = dict(
    data_path=DATA_PATH,
    window_size=30,
    strike_moneyness=1.0,
    rate=0.01,
    transaction_cost=0.001,
    augment_vol=True,       # vol scaling for extra training variety
)

N_ENVS           = 8
TOTAL_TIMESTEPS  = 500_000
EVAL_FREQ        = 10_000
N_EVAL_EPISODES  = 100

MODEL_PATH      = "models/ppo_hedger"
CHECKPOINT_DIR  = "models/ppo_checkpoints"
LOG_DIR         = "results/learning_curves/ppo"
PROGRESS_PATH   = os.path.join(ROOT, LOG_DIR, "progress.json")


class ProgressFileCallback(BaseCallback):
    """Writes timestep + training phase to a JSON file for live UI monitoring.

    Writes on:
      • every `write_every` env steps (during rollout collection)
      • _on_rollout_start  → phase = "collecting"
      • _on_rollout_end    → phase = "updating"  (fires before gradient updates)
    This ensures the file is always current; the UI can show "Updating policy…"
    during the gradient-update phase when _on_step is not called.
    """

    def __init__(self, path: str, total_steps: int, write_every: int = 2_000):
        super().__init__(verbose=0)
        self._path = path
        self._total = total_steps
        self._every = write_every
        self._next_write = write_every
        self._phase = "starting"

    def _write(self) -> None:
        # Write to a temp file then atomically rename so the reader never sees
        # a truncated / partially-written JSON (which would cause json.load to
        # fail and the UI to fall back to stale NPZ data, making the bar drop).
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "timesteps": int(self.num_timesteps),
                        "total":     self._total,
                        "phase":     self._phase,
                    },
                    fh,
                )
            os.replace(tmp, self._path)  # atomic on POSIX
        except OSError:
            pass

    def _on_training_start(self) -> None:
        self._phase = "starting"
        self._write()

    def _on_rollout_start(self) -> None:
        self._phase = "collecting"
        self._write()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_write:
            self._next_write = self.num_timesteps + self._every
            self._write()
        return True

    def _on_rollout_end(self) -> None:
        self._phase = "updating"
        self._write()  # always write before gradient updates start


def make_env(seed: int | None = None, extra: dict | None = None):
    kwargs = {**ENV_KWARGS, **(extra or {})}

    def _init():
        return RealDataHedgingEnv(**kwargs, seed=seed)
    return _init


def parse_args():
    ap = argparse.ArgumentParser(description="Train PPO hedging agent on real SPY data.")
    ap.add_argument(
        "--risk-objective", choices=["quadratic", "cvar"], default="quadratic",
        help="Reward objective. 'quadratic' minimises mean squared hedging error "
             "(tail-agnostic); 'cvar' adds a Rockafellar-Uryasev tail penalty.",
    )
    ap.add_argument("--cvar-alpha", type=float, default=0.95,
                    help="CVaR confidence level; 0.95 targets the worst 5%% of episodes.")
    ap.add_argument("--cvar-weight", type=float, default=1.0,
                    help="Weight on the CVaR term relative to squared-error terms.")
    ap.add_argument(
        "--split", choices=["all", "train", "test"], default="all",
        help="Temporal window split. 'all' uses every window (regime coverage, but "
             "leaks across time). 'train' holds out the most recent period with an "
             "embargo, so results on 'test' are genuinely out-of-sample.",
    )
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="Fraction of history assigned to the training split.")
    ap.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    ap.add_argument("--model-path", default=MODEL_PATH,
                    help="Output path without the .zip extension.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    TOTAL_TIMESTEPS = args.timesteps
    MODEL_PATH = args.model_path
    # Derive run-specific output dirs from the model name so that training a
    # variant (e.g. --model-path models/ppo_hedger_cvar) does not overwrite the
    # baseline run's learning curves, checkpoints or progress file.
    RUN_NAME = os.path.basename(MODEL_PATH)
    if RUN_NAME != "ppo_hedger":
        LOG_DIR = f"results/learning_curves/{RUN_NAME}"
        CHECKPOINT_DIR = f"models/{RUN_NAME}_checkpoints"
        PROGRESS_PATH = os.path.join(ROOT, LOG_DIR, "progress.json")
    ENV_EXTRA = dict(
        risk_objective=args.risk_objective,
        cvar_alpha=args.cvar_alpha,
        cvar_weight=args.cvar_weight,
        split=args.split,
        train_frac=args.train_frac,
    )
    if not os.path.exists(DATA_PATH):
        print(
            f"\nERROR: {DATA_PATH} not found.\n"
            "Run  python data/generate_data.py  first to fetch SPY history."
        )
        sys.exit(1)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Clear stale progress file so the UI starts from 0
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)

    # Report how many real windows are available
    _probe = RealDataHedgingEnv(**ENV_KWARGS, **ENV_EXTRA)
    print(f"\n  Real data windows available: {_probe.n_windows:,}  (split={args.split})")
    print(f"  Reward objective: {_probe.risk!r}")
    del _probe

    train_env = DummyVecEnv([make_env(seed=i, extra=ENV_EXTRA) for i in range(N_ENVS)])
    eval_env  = DummyVecEnv([make_env(seed=999, extra=ENV_EXTRA)])

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=f"{MODEL_PATH}_best",
            log_path=LOG_DIR,
            eval_freq=max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(50_000 // N_ENVS, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="ppo",
            verbose=0,
        ),
        ProgressFileCallback(PROGRESS_PATH, TOTAL_TIMESTEPS, write_every=2_000),
    ])

    existing = f"{MODEL_PATH}.zip"
    if os.path.exists(existing):
        print("\nLoading existing PPO model — continuing training on real SPY data…")
        model = PPO.load(existing, env=train_env, verbose=1)
        model.tensorboard_log = None  # prevent crash if tensorboard not installed
        # NOTE: do NOT overwrite model.clip_range — SB3 stores it as a callable
        # schedule; replacing with a float causes TypeError in ppo.train().
        model.learning_rate = 5e-5    # lower LR for fine-tuning
    else:
        print("\nNo existing model — training PPO from scratch on real SPY data…")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [256, 256]},
        )

    print(f"\nTraining PPO for {TOTAL_TIMESTEPS:,} steps across {N_ENVS} envs…")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=False,  # ProgressFileCallback handles UI updates instead
        reset_num_timesteps=True,
    )
    model.save(MODEL_PATH)

    train_env.close()
    eval_env.close()
    print(f"\nPPO training complete.  Model saved to '{MODEL_PATH}.zip'")
    print(f"Learning curves saved to '{LOG_DIR}/evaluations.npz'")
