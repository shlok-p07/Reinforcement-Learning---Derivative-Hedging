"""
SAC training on real SPY market data.

SAC's off-policy replay buffer naturally handles the temporal structure
of real market windows — it stores transitions from many different historical
periods and learns from them in random mini-batches, which reduces
overfitting to any single market regime.

Continuation training
---------------------
If models/sac_hedger.zip already exists the script loads it and continues
training.  Delete the file to restart from scratch.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from envs.real_data_env import RealDataHedgingEnv

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

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
    ])

    existing = f"{MODEL_PATH}.zip"
    if os.path.exists(existing):
        print(f"\nLoading existing SAC model — continuing training on real SPY data…")
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
        progress_bar=True,
        reset_num_timesteps=False,
    )
    model.save(MODEL_PATH)

    train_env.close()
    eval_env.close()
    print(f"\nSAC training complete.  Model saved to '{MODEL_PATH}.zip'")
    print(f"Learning curves saved to '{LOG_DIR}/evaluations.npz'")
