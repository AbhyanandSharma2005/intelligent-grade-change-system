"""Windowed features: rolling mean + slope of each tag over the last
N steps, computable both offline (training) and mid-transition (live)."""
import numpy as np
import pandas as pd

from src.config import PROCESS_TAGS, QUALITY_TAG, TRIGGER_STEP

WINDOW = 36  # 3 minutes at 5 s sampling


def _slope(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(x)), x, 1)[0])


def episode_features(ep: pd.DataFrame, at_step: int | None = None) -> dict:
    """Feature vector from data up to `at_step` (default: shortly after
    the ramp starts — the point where a live prediction is most useful)."""
    if at_step is None:
        at_step = TRIGGER_STEP + WINDOW
    w = ep.iloc[max(0, at_step - WINDOW):at_step]
    feats = {}
    for tag in PROCESS_TAGS + [QUALITY_TAG]:
        feats[f"{tag}_mean"] = float(w[tag].mean())
        feats[f"{tag}_slope"] = _slope(w[tag].values)
    feats["bw_dev_pct"] = float(
        abs(w[QUALITY_TAG].iloc[-1] - w["bw_setpoint"].iloc[-1])
        / w["bw_setpoint"].iloc[-1] * 100)
    feats["bw_setpoint_delta"] = float(
        ep["bw_setpoint"].iloc[-1] - ep["bw_setpoint"].iloc[0])
    return feats


def build_training_table(episodes: pd.DataFrame,
                         meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eid, ep in episodes.groupby("episode_id"):
        f = episode_features(ep.reset_index(drop=True))
        f["episode_id"] = eid
        rows.append(f)
    X = pd.DataFrame(rows).set_index("episode_id")
    y = meta.set_index("episode_id").loc[X.index, "off_spec"].astype(int)
    X["label_off_spec"] = y
    return X
