"""
src/forecast.py

Quantile deviation forecast: predicts basis-weight deviation (%) at fixed
horizons ahead, as P10/P50/P90, from the same windowed features the risk
model uses (src/features.py). This is the "future state if current trend
continues" deliverable, with real uncertainty bands rather than a point
estimate.

Called from api/deps.py (load_forecast() at startup) and api/main.py
(forecast_deviation(state.forecast_bundle, feats) per request).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR


def load_forecast(model_dir: str = MODEL_DIR):
    """Returns None if scripts/train_forecast.py hasn't been run yet.
    api/main.py's /forecast route already handles None with a 503
    ('Forecast model not trained') — this must not raise."""
    path = Path(model_dir) / "forecast.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def forecast_deviation(bundle: dict, feats: dict) -> list[dict]:
    """Returns [{horizon_s, dev_pct_p10, dev_pct_p50, dev_pct_p90}, ...].
    Key names match api/main.py's `p["dev_pct_p50"] > DEVIATION_LIMIT_PCT` check."""
    X = pd.DataFrame([feats])[bundle["feature_cols"]]
    points = []
    for h in bundle["horizons"]:
        preds = np.atleast_1d(bundle["models"][h].predict(X)[0])
        point = {"horizon_s": h}
        for q, val in zip(bundle["quantiles"], preds):
            point[f"dev_pct_p{int(q * 100)}"] = float(val)
        points.append(point)
    return points