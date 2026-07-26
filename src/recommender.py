"""k-NN setpoint recommendation + rationale generation with tags.

Feedback loop: episodes whose recommendations keep getting rejected are
down-weighted in retrieval — we widen the neighbor pool, inflate distances
for poorly-trusted episodes, then re-rank and keep the top-k. If no feedback
table exists yet (fresh install) this degrades gracefully to plain k-NN.
"""
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import (ACTUATOR_TAGS, CONFIGURED_LOOPS, QUALITY_TAG,
                        RECIPE_LIMITS, TRIGGER_STEP)

try:
    from src.config import FEEDBACK_DB_PATH
except ImportError:
    FEEDBACK_DB_PATH = "data/feedback.db"  # adjust if your config uses a different name/path

TAG_HISTORICAL = "[Historical Data]"
TAG_RECIPE = "[Recipe Limit]"
TAG_CORRELATION = "[New Correlation]"

FEEDBACK_SMOOTHING = 2.0  # pulls weight toward 0.5 until enough history accumulates
POOL_MULTIPLIER = 4       # widen the k-NN pool this many times before re-ranking by trust


@dataclass
class Recommendation:
    tag: str
    value: float
    primary_tag: str
    reason: str


def _load_episode_weights(db_path: str = FEEDBACK_DB_PATH,
                           smoothing: float = FEEDBACK_SMOOTHING) -> dict:
    """
    weight = (accepts + smoothing) / (accepts + rejects + 2*smoothing)
    Starts at 0.5 with no history; drifts toward 1.0 (trusted) as accepts
    accumulate, toward 0.0 (down-weighted) as rejects accumulate. Returns {}
    if the feedback table doesn't exist yet or the db is empty/missing —
    callers should treat a missing key as neutral (weight 0.5).
    """
    if not Path(db_path).exists():
        return {}
    try:
        con = sqlite3.connect(db_path)
        rows = pd.read_sql(
            "SELECT episode_id, accepted, COUNT(*) as n FROM feedback "
            "GROUP BY episode_id, accepted", con)
        con.close()
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return {}

    weights = {}
    for eid, grp in rows.groupby("episode_id"):
        accepts = int(grp.loc[grp.accepted == 1, "n"].sum())
        rejects = int(grp.loc[grp.accepted == 0, "n"].sum())
        weights[int(eid)] = (accepts + smoothing) / (accepts + rejects + 2 * smoothing)
    return weights


class SetpointRecommender:
    """Retrieves the most similar historical in-spec, fast-stabilizing
    episodes and suggests the setpoints they used."""

    def __init__(self, feature_table: pd.DataFrame, meta: pd.DataFrame,
                 episodes: pd.DataFrame, k: int = 5):
        good = meta[~meta["off_spec"]].copy()
        good = good.nsmallest(max(k * 8, 40), "time_to_stabilize_s")
        self.meta = good.set_index("episode_id")
        self.X = feature_table.drop(columns=["label_off_spec"]).loc[
            self.meta.index]
        self.scaler = StandardScaler().fit(self.X.values)
        self.knn = NearestNeighbors(n_neighbors=k).fit(
            self.scaler.transform(self.X.values))
        self.episodes = episodes
        self.k = k

    def _neighbor_setpoints(self, eid: int) -> dict:
        ep = self.episodes[self.episodes["episode_id"] == eid]
        stab = int(self.meta.loc[eid, "stabilization_step"])
        w = ep.iloc[stab:stab + 24]
        return {a: float(w[a].mean()) for a in ACTUATOR_TAGS}

    def _feedback_weighted_neighbors(self, x_scaled: np.ndarray) -> list[int]:
        """Widen the k-NN pool, inflate distances for low-trust episodes,
        re-sort, and return the top-k episode_ids."""
        pool_size = min(len(self.X), self.k * POOL_MULTIPLIER)
        dist, idx = self.knn.kneighbors(x_scaled, n_neighbors=pool_size)
        candidate_ids = self.X.index[idx[0]].to_numpy()
        distances = dist[0]

        weights = _load_episode_weights()
        if not weights:
            return candidate_ids[: self.k].tolist()

        penalty = np.array([1.0 / max(weights.get(int(eid), 0.5), 0.05)
                             for eid in candidate_ids])
        penalized = distances * penalty
        order = np.argsort(penalized)
        return candidate_ids[order][: self.k].tolist()

    def recommend(self, live_feats: dict,
                  new_corr_tags: list[str]) -> list[Recommendation]:
        x = pd.DataFrame([live_feats])[self.X.columns].values
        x_scaled = self.scaler.transform(x)
        neighbor_ids = self._feedback_weighted_neighbors(x_scaled)

        sps = pd.DataFrame([self._neighbor_setpoints(e) for e in neighbor_ids])
        recs = []
        for a in ACTUATOR_TAGS:
            raw = float(sps[a].median())
            lo, hi = RECIPE_LIMITS[a]
            clipped = float(np.clip(raw, lo, hi))
            n = len(neighbor_ids)
            avg_stab = self.meta.loc[neighbor_ids, "time_to_stabilize_s"].mean()

            if clipped != raw:
                tag, reason = TAG_RECIPE, (
                    f"{a} suggestion {raw:.0f} clipped to recipe limit "
                    f"[{lo:.0f}, {hi:.0f}] → {clipped:.0f}.")
            elif (a not in CONFIGURED_LOOPS.get(QUALITY_TAG, [])
                  and any(t in new_corr_tags for t in ["moisture",
                                                       "dryer_humidity"])
                  and a == "steam_pressure"):
                tag, reason = TAG_CORRELATION, (
                    f"steam_pressure at {clipped:.0f} — a newly discovered "
                    f"correlation (dryer humidity → moisture → basis weight) "
                    f"is not compensated by any configured loop; adjusting "
                    f"steam counteracts it.")
            else:
                tag, reason = TAG_HISTORICAL, (
                    f"{a} recommended at {clipped:.0f} — similar to {n} past "
                    f"transitions that stayed in-spec and stabilized in "
                    f"~{avg_stab:.0f}s on average.")
            recs.append(Recommendation(a, clipped, tag, reason))
        return recs, neighbor_ids