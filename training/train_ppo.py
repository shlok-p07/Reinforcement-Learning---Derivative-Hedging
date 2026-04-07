"""
PPO training on real SPY market data.

The agent trains by replaying random 30-day windows sampled from 5 years
of real SPY daily price history.  Each episode the environment picks a
different historical window, so the agent trains on genuine market dynamics
(fat tails, vol clustering, crashes, rallies) rather than synthetic GBM.

~1,200 distinct 30-day windows available from 5 years of SPY data.
With 8 parallel envs each independently sampling, the agent sees highly
varied market conditions throughout training.

Continuation training
---------------------
If models/ppo_hedger.zip already exists the script loads it and continues
training from its current weights (lower LR for fine-tuning).  Delete the
file to restart from scratch.

Hyperparameter rationale
------------------------
n_steps=2048       shorter rollouts suit the 30-step episode structure
batch_size=256     stable gradient estimates
n_epochs=10        standard PPO
gamma=0.99         near-undiscounted (30-step episodes)
gae_lambda=0.95    standard GAE for variance reduction
clip_range=0.2     conservative clipping
ent_coef=0.005     small entropy bonus prevents premature policy collapse
learning_rate=1e-4 lower than default; option P&L rewards have high variance
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stable_baselines3 import PPO
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
    augment_vol=True,       # vol scaling for extra training variety
)

N_ENVS           = 8
TOTAL_TIMESTEPS  = 500_000
EVAL_FREQ        = 10_000
N_EVAL_EPISODES  = 100

MODEL_PATH      = "models/ppo_hedger"
CHECKPOINT_DIR  = "models/ppo_checkpoints"
LOG_DIR         = "results/learning_curves/ppo"


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
    ])

    existing = f"{MODEL_PATH}.zip"
    if os.path.exists(existing):
        print(f"\nLoading existing PPO model — continuing training on real SPY data…")
        model = PPO.load(existing, env=train_env, verbose=1)
        model.clip_range   = 0.2
        model.learning_rate = 5e-5   # lower LR for fine-tuning
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
        progress_bar=True,
        reset_num_timesteps=False,
    )
    model.save(MODEL_PATH)

    train_env.close()
    eval_env.close()
    print(f"\nPPO training complete.  Model saved to '{MODEL_PATH}.zip'")
    print(f"Learning curves saved to '{LOG_DIR}/evaluations.npz'")
