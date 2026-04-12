"""
PPO training on real SPY market data (~1,200 distinct 30-day windows, 5-year history).

Loads and continues from models/ppo_hedger.zip if it exists; delete to retrain from scratch.
Hyperparams: n_steps=2048, batch_size=256, n_epochs=10, γ=0.99, λ=0.95, clip=0.2,
             ent_coef=0.005, lr=1e-4 (reduced to 5e-5 when fine-tuning).
"""

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
    """Writes current timestep to a JSON file every N steps for live UI monitoring."""

    def __init__(self, path: str, total_steps: int, write_every: int = 2_000):
        super().__init__(verbose=0)
        self._path = path
        self._total = total_steps
        self._every = write_every
        self._next_write = write_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_write:
            self._next_write = self.num_timesteps + self._every
            try:
                with open(self._path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {"timesteps": int(self.num_timesteps), "total": self._total}, fh
                    )
            except OSError:
                pass
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

    # Report how many real windows are available
    _probe = RealDataHedgingEnv(**ENV_KWARGS)
    print(f"\n  Real data windows available: {_probe.n_windows:,}")
    del _probe

    train_env = DummyVecEnv([make_env(seed=i) for i in range(N_ENVS)])
    eval_env  = DummyVecEnv([make_env(seed=999)])

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
        model.tensorboard_log = None   # prevent crash if tensorboard not installed
        model.clip_range      = 0.2
        model.learning_rate   = 5e-5  # lower LR for fine-tuning
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
