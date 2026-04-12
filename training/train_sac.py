"""
SAC training on real SPY market data (~1,200 distinct 30-day windows, 5-year history).

Loads and continues from models/sac_hedger.zip if it exists; delete to retrain from scratch.
Hyperparams: lr=3e-4, buffer=200k, batch=256, γ=0.99, τ=0.005, ent_coef=auto, use_sde=True.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stable_baselines3 import SAC
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
    augment_vol=True,
)

TOTAL_TIMESTEPS = 300_000
EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 100

MODEL_PATH     = "models/sac_hedger"
CHECKPOINT_DIR = "models/sac_checkpoints"
LOG_DIR        = "results/learning_curves/sac"
PROGRESS_PATH  = os.path.join(ROOT, LOG_DIR, "progress.json")


class ProgressFileCallback(BaseCallback):
    """Writes timestep + training phase to a JSON file for live UI monitoring."""

    def __init__(self, path: str, total_steps: int, write_every: int = 2_000):
        super().__init__(verbose=0)
        self._path = path
        self._total = total_steps
        self._every = write_every
        self._next_write = write_every
        self._phase = "starting"

    def _write(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "timesteps": int(self.num_timesteps),
                        "total":     self._total,
                        "phase":     self._phase,
                    },
                    fh,
                )
        except OSError:
            pass

    def _on_training_start(self) -> None:
        self._phase = "collecting"
        self._write()

    def _on_step(self) -> bool:
        self._phase = "collecting"
        if self.num_timesteps >= self._next_write:
            self._next_write = self.num_timesteps + self._every
            self._write()
        return True


def make_env(seed: int | None = None):
    def _init():
        return RealDataHedgingEnv(**ENV_KWARGS, seed=seed)
    return _init


if __name__ == "__main__":
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

    _probe = RealDataHedgingEnv(**ENV_KWARGS)
    print(f"\n  Real data windows available: {_probe.n_windows:,}")
    del _probe

    # SAC is off-policy — two envs is sufficient
    train_env = DummyVecEnv([make_env(seed=i) for i in range(2)])
    eval_env  = DummyVecEnv([make_env(seed=999)])

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=f"{MODEL_PATH}_best",
            log_path=LOG_DIR,
            eval_freq=EVAL_FREQ // 2,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000 // 2,
            save_path=CHECKPOINT_DIR,
            name_prefix="sac",
            verbose=0,
        ),
        ProgressFileCallback(PROGRESS_PATH, TOTAL_TIMESTEPS, write_every=2_000),
    ])

    existing = f"{MODEL_PATH}.zip"
    if os.path.exists(existing):
        print("\nLoading existing SAC model — continuing training on real SPY data…")
        model = SAC.load(existing, env=train_env, verbose=1)
        model.learning_rate = 1e-4
    else:
        print("\nNo existing model — training SAC from scratch on real SPY data…")
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=200_000,
            learning_starts=2_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            use_sde=True,
            sde_sample_freq=64,
            policy_kwargs={"net_arch": [256, 256]},
        )

    print(f"\nTraining SAC for {TOTAL_TIMESTEPS:,} steps…")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=False,  # ProgressFileCallback handles UI updates instead
        reset_num_timesteps=True,
    )
    model.save(MODEL_PATH)

    train_env.close()
    eval_env.close()
    print(f"\nSAC training complete.  Model saved to '{MODEL_PATH}.zip'")
    print(f"Learning curves saved to '{LOG_DIR}/evaluations.npz'")
