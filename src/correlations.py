"""Correlation discovery: which tags actually drive basis weight,
diffed against the loops already configured in the QCS."""
import numpy as np
import pandas as pd

from src.config import CONFIGURED_LOOPS, PROCESS_TAGS, QUALITY_TAG


def cross_correlations(episodes: pd.DataFrame, max_lag: int = 24) -> pd.DataFrame:
    """Max |lagged cross-correlation| of each tag vs basis weight,
    averaged across episodes (on first-differenced signals)."""
    rows = []
    for _, ep in episodes.groupby("episode_id"):
        bw = np.diff(ep[QUALITY_TAG].values)
        for tag in PROCESS_TAGS:
            x = np.diff(ep[tag].values)
            best = 0.0
            for lag in range(0, max_lag + 1):
                a = x[:len(x) - lag] if lag else x
                b = bw[lag:] if lag else bw
                if a.std() < 1e-9 or b.std() < 1e-9:
                    continue
                c = abs(np.corrcoef(a, b)[0, 1])
                best = max(best, c)
            rows.append({"tag": tag, "corr": best})
    return (pd.DataFrame(rows).groupby("tag")["corr"].mean()
            .sort_values(ascending=False).reset_index())


def discover_new_correlations(episodes: pd.DataFrame,
                              shap_drivers: list[tuple[str, float]],
                              corr_threshold: float = 0.15) -> pd.DataFrame:
    """Tags correlated with basis weight (or flagged by SHAP) that are
    NOT among the configured control-loop inputs for basis weight."""
    known = set(CONFIGURED_LOOPS.get(QUALITY_TAG, []))
    corr = cross_correlations(episodes)
    shap_tags = {f.rsplit("_", 1)[0] for f, _ in shap_drivers}

    out = []
    for _, r in corr.iterrows():
        tag = r["tag"]
        if tag in known or tag == QUALITY_TAG:
            continue
        flagged_by_shap = tag in shap_tags
        if r["corr"] >= corr_threshold or flagged_by_shap:
            out.append({"tag": tag, "avg_cross_corr": round(r["corr"], 3),
                        "flagged_by_shap": flagged_by_shap,
                        "in_configured_loops": False})
    return pd.DataFrame(out)
