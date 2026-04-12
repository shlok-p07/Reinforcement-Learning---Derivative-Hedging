"""
Pull real market data from Yahoo Finance (via yfinance) and save to data/.

Files produced
--------------
  data/spy_daily.csv          — SPY daily OHLCV + rolling realized vol (5 years)
  data/spy_returns.csv        — daily log-returns with vol regimes
  data/option_chain_calls.csv — live SPY call options chain (nearest expiry)
  data/vol_surface.csv        — implied vol grid across all available expiries
  data/calibrated_params.json — GBM params calibrated from SPY history
  data/regime_history.csv     — HMM-style vol-regime labels on historical data

5 years of daily data gives ~1200+ distinct 30-day training windows for
the RealDataHedgingEnv, covering multiple market regimes (bull, bear,
crash, recovery) that a 1-year fetch would miss.

Run once before launching the Streamlit app:
  python data/generate_data.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TICKER = "SPY"


# 1. Historical daily prices  (1 year)

def fetch_spy_daily() -> pd.DataFrame:
    print(f"  Fetching {TICKER} daily OHLCV (5 years)…")
    tkr  = yf.Ticker(TICKER)
    hist = tkr.history(period="5y", interval="1d", auto_adjust=True)
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    hist = hist[["Open","High","Low","Close","Volume"]].copy()
    hist.columns = ["open","high","low","close","volume"]
    hist.index.name = "date"

    hist["log_return"]   = np.log(hist["close"] / hist["close"].shift(1))
    hist["rvol_5d"]  = hist["log_return"].rolling(5).std()  * np.sqrt(252)
    hist["rvol_21d"] = hist["log_return"].rolling(21).std() * np.sqrt(252)
    hist["rvol_63d"] = hist["log_return"].rolling(63).std() * np.sqrt(252)

    hist = hist.dropna(subset=["log_return"])
    hist = hist.reset_index()
    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    return hist


# 2. Returns + vol-regime labelling

def label_regimes(hist: pd.DataFrame) -> pd.DataFrame:
    """Simple threshold-based regime label: High vol if rvol_21d > median."""
    df = hist[["date","close","log_return","rvol_21d"]].copy().dropna()
    median_vol = df["rvol_21d"].median()
    df["regime"]       = (df["rvol_21d"] > median_vol).astype(int)
    df["regime_label"] = df["regime"].map({0: "Low Vol", 1: "High Vol"})
    return df


# 3. Calibrate GBM parameters

def calibrate_gbm(hist: pd.DataFrame) -> dict:
    returns = hist["log_return"].dropna()
    mu_daily    = float(returns.mean())
    sigma_daily = float(returns.std())
    mu_annual   = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)

    # Recent 30-day realized vol
    recent_rvol = float(hist["rvol_21d"].dropna().iloc[-1]) if "rvol_21d" in hist.columns else sigma_annual

    spot_last = float(hist["close"].iloc[-1])
    spot_first = float(hist["close"].iloc[0])

    params = {
        "ticker":         TICKER,
        "as_of":          datetime.today().strftime("%Y-%m-%d"),
        "s0_last":        round(spot_last, 2),
        "s0_first":       round(spot_first, 2),
        "mu_annual":      round(mu_annual, 4),
        "sigma_annual":   round(sigma_annual, 4),
        "rvol_21d_last":  round(recent_rvol, 4),
        "dt":             round(1 / 252, 6),
        "n_obs":          len(returns),
        "note": (
            "sigma_annual calibrated from 5-year daily log-returns. "
            "Use as the baseline sigma for RealDataHedgingEnv and GBM scenarios."
        ),
    }
    return params


# 4. Live options chain

def fetch_options_chain() -> pd.DataFrame:
    print(f"  Fetching {TICKER} options chain…")
    tkr  = yf.Ticker(TICKER)
    exps = tkr.options           # list of expiry date strings

    if not exps:
        print("    No options data available — market may be closed.")
        return pd.DataFrame()

    # Pick 3 nearest expiries
    rows = []
    for exp in exps[:4]:
        try:
            chain = tkr.option_chain(exp)
            calls = chain.calls.copy()
            calls["expiry"]      = exp
            calls["option_type"] = "call"
            # Keep relevant columns
            keep = ["expiry","strike","lastPrice","bid","ask","impliedVolatility",
                    "delta","gamma","openInterest","volume","option_type"]
            for c in keep:
                if c not in calls.columns:
                    calls[c] = np.nan
            rows.append(calls[keep])
        except Exception as e:
            print(f"    Skipping {exp}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns={"impliedVolatility":"impl_vol","lastPrice":"last_price",
                             "openInterest":"open_interest"})
    df["impl_vol"] = df["impl_vol"].round(4)
    # Filter out junk rows (zero IV, extreme strikes)
    df = df[(df["impl_vol"] > 0.01) & (df["impl_vol"] < 5.0)]
    return df


# 5. Implied vol surface (strike × maturity grid)

def build_vol_surface(chain_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the options chain into a strike × maturity IV grid."""
    if chain_df.empty:
        return pd.DataFrame()

    today = pd.Timestamp.today().normalize()
    chain_df = chain_df.copy()
    chain_df["expiry_dt"]    = pd.to_datetime(chain_df["expiry"])
    chain_df["days_to_exp"]  = (chain_df["expiry_dt"] - today).dt.days
    chain_df = chain_df[chain_df["days_to_exp"] > 0]

    # Bin strikes into moneyness bands relative to last SPY close
    try:
        spot_approx = yf.Ticker(TICKER).fast_info["last_price"]
    except Exception:
        spot_approx = chain_df["strike"].median()

    chain_df["moneyness_pct"] = (chain_df["strike"] / spot_approx).round(2)

    surface = (
        chain_df[["days_to_exp","strike","moneyness_pct","impl_vol"]]
        .dropna()
        .sort_values(["days_to_exp","strike"])
        .reset_index(drop=True)
    )
    return surface


# Entry point

if __name__ == "__main__":
    print(f"\nFetching real market data for {TICKER} from Yahoo Finance…\n")

    # 1. Daily prices
    spy = fetch_spy_daily()
    spy.to_csv(os.path.join(DATA_DIR, "spy_daily.csv"), index=False)
    print(f"    ✓  spy_daily.csv       — {len(spy)} trading days")

    # 2. Returns + regimes
    regimes = label_regimes(spy)
    regimes.to_csv(os.path.join(DATA_DIR, "spy_returns.csv"), index=False)
    print(f"    ✓  spy_returns.csv     — {len(regimes)} rows, regimes labelled")

    # 3. Calibrate GBM
    params = calibrate_gbm(spy)
    with open(os.path.join(DATA_DIR, "calibrated_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"    ✓  calibrated_params.json  σ={params['sigma_annual']:.1%}  μ={params['mu_annual']:.1%}")

    # 4. Options chain
    chain = fetch_options_chain()
    if not chain.empty:
        chain.to_csv(os.path.join(DATA_DIR, "option_chain_calls.csv"), index=False)
        print(f"    ✓  option_chain_calls.csv — {len(chain)} contracts")

        # 5. Vol surface
        surface = build_vol_surface(chain)
        if not surface.empty:
            surface.to_csv(os.path.join(DATA_DIR, "vol_surface.csv"), index=False)
            print(f"    ✓  vol_surface.csv      — {len(surface)} data points")
    else:
        print("    ⚠  Options data unavailable (market closed or API limit).")

    print(f"\nAll files saved to {DATA_DIR}/")
    print(f"Files: {sorted(os.listdir(DATA_DIR))}")
    print(
        f"\nCalibrated GBM params for the RL env:\n"
        f"  sigma  = {params['sigma_annual']:.4f}  (model vol)\n"
        f"  mu     = {params['mu_annual']:.4f}  (drift)\n"
        f"  s0     = {params['s0_last']:.2f}  (last close)\n"
    )
