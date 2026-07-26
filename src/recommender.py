"""k-NN setpoint recommendation + rationale generation with tags."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import (ACTUATOR_TAGS, CONFIGURED_LOOPS, QUALITY_TAG,
                        RECIPE_LIMITS, TRIGGER_STEP)

TAG_HISTORICAL = "[Historical Data]"
TAG_RECIPE = "[Recipe Limit]"
TAG_CORRELATION = "[New Correlation]"


@dataclass
class Recommendation:
    tag: str
    value: float
    primary_tag: str
    reason: str


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

    def recommend(self, live_feats: dict,
                  new_corr_tags: list[str]) -> list[Recommendation]:
        x = pd.DataFrame([live_feats])[self.X.columns].values
        _, idx = self.knn.kneighbors(self.scaler.transform(x))
        neighbor_ids = self.X.index[idx[0]].tolist()

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
