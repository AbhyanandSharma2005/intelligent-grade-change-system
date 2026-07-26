"""Quantile forecast of basis-weight deviation: 'future state if the
current trend continues', with uncertainty bands (P10/P50/P90)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (MODEL_DIR, SAMPLE_SECONDS, TRIGGER_STEP)
from src.features import episode_features

HORIZONS = [12, 24, 36]          # steps ahead: 1, 2, 3 minutes
QUANTILES = [0.1, 0.5, 0.9]
FORECAST_PATH = Path(MODEL_DIR) / "forecast.joblib"


def build_forecast_dataset(episodes: pd.DataFrame):
    """Rows: features at step t -> bw deviation %% at t + h."""
    X_rows, y = [], {h: [] for h in HORIZONS}
    max_h = max(HORIZONS)
    for _, ep in episodes.groupby("episode_id"):
        ep = ep.reset_index(drop=True)
        dev = ((ep["basis_weight"] - ep["bw_setpoint"]).abs()
               / ep["bw_setpoint"] * 100).values
        for t in range(TRIGGER_STEP, len(ep) - max_h, 8):
            X_rows.append(episode_features(ep, at_step=t))
            for h in HORIZONS:
                y[h].append(dev[t + h])
    X = pd.DataFrame(X_rows)
    return X, {h: np.array(v) for h, v in y.items()}


def train_forecast(episodes: pd.DataFrame) -> dict:
    X, targets = build_forecast_dataset(episodes)
    models = {}
    for h in HORIZONS:
        for q in QUANTILES:
            m = xgb.XGBRegressor(objective="reg:quantileerror",
                                 quantile_alpha=q, n_estimators=150,
                                 max_depth=4, learning_rate=0.1)
            m.fit(X, targets[h])
            models[(h, q)] = m
    bundle = {"models": models, "columns": list(X.columns)}
    joblib.dump(bundle, FORECAST_PATH)
    return bundle


def load_forecast():
    return joblib.load(FORECAST_PATH) if FORECAST_PATH.exists() else None


def forecast_deviation(fbundle, feats: dict) -> list[dict]:
    """Predicted future bw deviation %% (P10/P50/P90) per horizon."""
    X = pd.DataFrame([feats])[fbundle["columns"]]
    out = []
    for h in HORIZONS:
        q10, q50, q90 = (float(fbundle["models"][(h, q)].predict(X)[0])
                         for q in QUANTILES)
        # quantile crossing guard
        q10, q50, q90 = sorted([max(0, q10), max(0, q50), max(0, q90)])
        out.append({"horizon_s": h * SAMPLE_SECONDS,
                    "dev_pct_p10": round(q10, 2),
                    "dev_pct_p50": round(q50, 2),
                    "dev_pct_p90": round(q90, 2)})
    return out
