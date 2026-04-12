"""Episode runner — collects step-by-step data for the live demo and evaluation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from envs.rl_hedging_env  import RLHedgingEnv       # noqa: E402
from envs.real_data_env   import RealDataHedgingEnv  # noqa: E402


ROOT      = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_PATH = os.path.join(ROOT, "data", "spy_daily.csv")

# Presets with "data_path" route to RealDataHedgingEnv; others use RLHedgingEnv.
ENV_PRESETS: dict[str, dict] = {
    "SPY Historical (Real Data)": dict(
        data_path=DATA_PATH,
        window_size=30,
        strike_moneyness=1.0,
        rate=0.01,
        transaction_cost=0.001,
        augment_vol=False,          # pure historical replay — no vol scaling
    ),
    "Base Market": dict(
        s0=100.0, mu=0.05, sigma=0.20, dt=1/252, maturity=30/252,
        strike=100.0, rate=0.01, transaction_cost=0.001,
    ),
    "High Transaction Cost": dict(
        s0=100.0, mu=0.05, sigma=0.20, dt=1/252, maturity=30/252,
        strike=100.0, rate=0.01, transaction_cost=0.01,
    ),
    "Volatility Mismatch": dict(
        s0=100.0, mu=0.05, sigma=0.20, dt=1/252, maturity=30/252,
        strike=100.0, rate=0.01, transaction_cost=0.001, realized_sigma=0.30,
    ),
    "Regime Switching": dict(
        s0=100.0, mu=0.05, sigma=0.20, dt=1/252, maturity=30/252,
        strike=100.0, rate=0.01, transaction_cost=0.001,
        regime_switching=True, sigma_low=0.15, sigma_high=0.35,
    ),
}

# Keys the environment info dict must contain
_REQUIRED_INFO_KEYS = (
    "spot", "tau", "portfolio_value", "option_value",
    "delta_v", "total_tc", "regime", "realized_vol",
)


def _make_env(env_kwargs: dict, seed: int):
    """Construct the right environment type from kwargs."""
    if "data_path" in env_kwargs:
        return RealDataHedgingEnv(**env_kwargs, seed=seed)
    return RLHedgingEnv(**env_kwargs, seed=seed)


def _safe_delta(obs: np.ndarray) -> float:
    """Extract delta from observation vector, with shape guard."""
    obs = np.asarray(obs, dtype=float).ravel()
    return float(obs[2]) if len(obs) > 2 else 0.0


def _initial_row(env, env_kwargs: dict, obs: np.ndarray) -> dict:
    """Build the step-0 row, compatible with both env types."""
    spot   = float(getattr(env, "spot",   100.0))
    tau    = float(getattr(env, "tau",    0.0))
    strike = float(getattr(env, "strike", env_kwargs.get("strike", 100.0)))
    sigma  = float(getattr(env, "sigma",  env_kwargs.get("sigma", 0.2)))

    option_val = 0.0
    if hasattr(env, "option") and env.option is not None:
        try:
            option_val = float(env.option.price(spot, max(tau, 1e-8)))
        except Exception:
            pass

    # realized vol — real env stores on self.sigma; GBM env on simulator
    sim = getattr(env, "simulator", None)
    if sim is not None:
        realized_vol = float(getattr(sim, "realized_sigma", sigma))
    else:
        realized_vol = sigma

    return {
        "step":            0,
        "spot":            spot,
        "tau":             tau,
        "delta":           _safe_delta(obs),
        "hedge_position":  0.0,
        "portfolio_value": 0.0,
        "option_value":    option_val,
        "delta_v":         0.0,
        "cumulative_tc":   0.0,
        "regime":          0,
        "realized_vol":    realized_vol,
        "strike":          strike,
    }


def collect_episode(
    env_kwargs: dict,
    strategy: str,
    model=None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run a full episode and return a DataFrame of per-step state.

    Works with both RLHedgingEnv (synthetic) and RealDataHedgingEnv.
    Returns an empty DataFrame on any unrecoverable error so callers
    can check `len(df) < 2` rather than catching exceptions.

    Columns: step, spot, tau, delta, hedge_position, portfolio_value,
             option_value, delta_v, cumulative_tc, regime, realized_vol, strike
    """
    try:
        env = _make_env(env_kwargs, seed)
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=float).ravel()
    except Exception:
        return pd.DataFrame()

    rows = [_initial_row(env, env_kwargs, obs)]

    done     = False
    step     = 0
    # max_steps: episode length derived from maturity/dt or window_size
    if "data_path" in env_kwargs:
        max_steps = env_kwargs.get("window_size", 30) + 5
    else:
        max_steps = int(
            env_kwargs.get("maturity", 30 / 252) / env_kwargs.get("dt", 1 / 252)
        ) + 10

    while not done and step < max_steps:
        step += 1

        try:
            if strategy == "no_hedge":
                action = np.array([0.0])
            elif strategy == "delta":
                action = np.array([_safe_delta(obs)])
            elif strategy == "random":
                action = np.array([np.random.uniform(-1.0, 1.0)])
            elif model is not None:
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=float).ravel()
            else:
                action = np.array([_safe_delta(obs)])   # fallback to delta
        except Exception:
            action = np.array([0.0])

        try:
            obs, _reward, terminated, truncated, info = env.step(action)
            obs  = np.asarray(obs, dtype=float).ravel()
            done = terminated or truncated
        except Exception:
            break

        rows.append({
            "step":            step,
            "spot":            float(info.get("spot",            rows[-1]["spot"])),
            "tau":             float(info.get("tau",             0.0)),
            "delta":           _safe_delta(obs),
            "hedge_position":  float(getattr(env, "stock_position", 0.0)),
            "portfolio_value": float(info.get("portfolio_value", 0.0)),
            "option_value":    float(info.get("option_value",    0.0)),
            "delta_v":         float(info.get("delta_v",         0.0)),
            "cumulative_tc":   float(info.get("total_tc",        0.0)),
            "regime":          int(info.get("regime",            0)),
            "realized_vol":    float(info.get("realized_vol",    rows[-1]["realized_vol"])),
            "strike":          float(getattr(env, "strike",      rows[-1]["strike"])),
        })

    return pd.DataFrame(rows)


def load_model(name: str):
    """Load a trained SB3 model. Returns None if not found or corrupted."""
    try:
        from stable_baselines3 import PPO, SAC  # noqa: PLC0415
    except ImportError:
        return None

    model_map = {"ppo": (PPO, "models/ppo_hedger"), "sac": (SAC, "models/sac_hedger")}
    if name not in model_map:
        return None

    cls, path = model_map[name]
    full_path  = os.path.join(ROOT, path)

    if not os.path.exists(f"{full_path}.zip"):
        return None

    try:
        model = cls.load(full_path)
        if not callable(getattr(model, "predict", None)):
            return None
        return model
    except Exception:
        return None


def quick_compare(
    env_kwargs: dict,
    strategies: list[str],
    models: dict,
    n_episodes: int = 100,
) -> dict[str, list[float]]:
    """
    Fast Monte Carlo comparison — returns {strategy: [pnl_list]}.
    Failed episodes are skipped silently.
    """
    results: dict[str, list[float]] = {s: [] for s in strategies}

    for seed in range(n_episodes):
        for strategy in strategies:
            ep_df = collect_episode(
                env_kwargs, strategy, model=models.get(strategy), seed=seed
            )
            if ep_df is not None and len(ep_df) >= 2:
                results[strategy].append(float(ep_df["portfolio_value"].iloc[-1]))

    return results
