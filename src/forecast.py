"""
scripts/train_forecast.py

Trains XGBoost quantile-regression models (P10/P50/P90) that forecast
basis-weight deviation (%) at fixed horizons ahead, using the same windowed
features as the risk model (src/features.py: rolling mean + slope per tag).
Saves models/forecast.joblib, loaded at API startup by src.forecast.load_forecast().

Requires xgboost>=2.0 (native 'reg:quantileerror' multi-quantile objective).
Check with: python -c "import xgboost; print(xgboost.__version__)"

Usage:
    python scripts/train_forecast.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import DATA_PATH, MODEL_DIR, SAMPLE_SECONDS, TRIGGER_STEP
from src.features import episode_features, WINDOW

HORIZONS_SEC = [30, 60, 120, 300]
QUANTILES = [0.1, 0.5, 0.9]
STRIDE_STEPS = 4  # sample every 4th step within an episode to build training rows


def _future_dev_pct(ep: pd.DataFrame, at_step: int):
    """Ground-truth target: bw deviation % at `at_step`, computed the same
    way features.py computes bw_dev_pct for the live/current step."""
    if at_step >= len(ep):
        return None
    row = ep.iloc[at_step]
    if row["bw_setpoint"] == 0:
        return None
    return float(abs(row["basis_weight"] - row["bw_setpoint"])
                / row["bw_setpoint"] * 100)


def build_forecast_table(episodes: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    rows = []
    for eid, ep in episodes.groupby("episode_id"):
        ep = ep.reset_index(drop=True)
        start = TRIGGER_STEP + WINDOW
        for at_step in range(start, len(ep) - horizon_steps, STRIDE_STEPS):
            target = _future_dev_pct(ep, at_step + horizon_steps)
            if target is None:
                continue
            feats = episode_features(ep, at_step=at_step)
            feats["target_dev_pct"] = target
            rows.append(feats)
    return pd.DataFrame(rows)


def train_one_horizon(df: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILES,
        n_estimators=250, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
    )
    model.fit(df[feature_cols], df["target_dev_pct"])
    return model


def main():
    episodes = pd.read_parquet(DATA_PATH)
    models, feature_cols = {}, None

    for h in HORIZONS_SEC:
        horizon_steps = max(1, round(h / SAMPLE_SECONDS))
        print(f"[forecast] building training rows for horizon={h}s "
             f"({horizon_steps} steps)...")
        df = build_forecast_table(episodes, horizon_steps)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != "target_dev_pct"]
        print(f"[forecast] training horizon={h}s on {len(df)} rows...")
        models[h] = train_one_horizon(df, feature_cols)

    bundle = {"models": models, "feature_cols": feature_cols,
             "quantiles": QUANTILES, "horizons": HORIZONS_SEC}
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, Path(MODEL_DIR) / "forecast.joblib")
    print(f"[forecast] saved -> {MODEL_DIR}/forecast.joblib")


if __name__ == "__main__":
    main()