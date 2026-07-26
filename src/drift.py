"""PSI-based feature drift detection against the training distribution."""
import numpy as np
import pandas as pd

PSI_WARN, PSI_ALERT = 0.1, 0.25


def psi(ref_quantiles: list[float], values: np.ndarray) -> float:
    """Population Stability Index using reference decile bins."""
    edges = np.unique(ref_quantiles)
    if len(edges) < 3 or len(values) == 0:
        return 0.0
    expected = np.full(len(edges) - 1, 1.0 / (len(edges) - 1))
    actual, _ = np.histogram(values, bins=edges)
    actual = actual / max(actual.sum(), 1)
    e, a = np.clip(expected, 1e-4, None), np.clip(actual, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def check_drift(bundle, recent_features: pd.DataFrame) -> pd.DataFrame:
    """Score each model feature; returns tags with status columns."""
    rows = []
    for col, ref_q in bundle["feature_ref_quantiles"].items():
        if col not in recent_features:
            continue
        score = psi(ref_q, recent_features[col].values)
        status = ("alert" if score >= PSI_ALERT
                  else "warn" if score >= PSI_WARN else "ok")
        rows.append({"feature": col, "psi": round(score, 4),
                     "status": status})
    return (pd.DataFrame(rows)
            .sort_values("psi", ascending=False).reset_index(drop=True))
