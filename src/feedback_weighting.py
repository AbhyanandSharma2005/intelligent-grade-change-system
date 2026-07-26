"""
Additions for src/recommender.py + src/feedback.py

Closes the feedback loop: every historical episode used as a k-NN neighbor
accumulates an accept/reject record. Episodes whose recommendations keep
getting rejected are down-weighted (pushed further away in retrieval ranking)
so the k-NN recommender naturally stops surfacing advice operators don't trust.
"""
import sqlite3
from typing import Dict
import numpy as np
import pandas as pd


def get_episode_weights(feedback_db_path: str, smoothing: float = 2.0) -> Dict[int, float]:
    """
    weight = (accepts + smoothing) / (accepts + rejects + 2*smoothing)

    Starts at 0.5 with no history, drifts toward 1.0 (trusted neighbor) as
    accepts accumulate, toward 0.0 (down-weighted) as rejects accumulate.
    `smoothing` prevents one early rejection from permanently blacklisting
    an episode — assumes a `feedback` table with columns
    (episode_id, accepted [0/1], timestamp). Adjust table/column names to
    match your existing feedback.py schema if different.
    """
    con = sqlite3.connect(feedback_db_path)
    rows = pd.read_sql(
        "SELECT episode_id, accepted, COUNT(*) as n FROM feedback "
        "GROUP BY episode_id, accepted", con,
    )
    con.close()

    weights: Dict[int, float] = {}
    for eid, grp in rows.groupby("episode_id"):
        accepts = int(grp.loc[grp.accepted == 1, "n"].sum())
        rejects = int(grp.loc[grp.accepted == 0, "n"].sum())
        weights[eid] = (accepts + smoothing) / (accepts + rejects + 2 * smoothing)
    return weights


def apply_feedback_penalty(distances: np.ndarray, neighbor_episode_ids: np.ndarray,
                            weights: Dict[int, float], default_weight: float = 0.5) -> np.ndarray:
    """
    Inflate raw k-NN distances for episodes with a poor track record, without
    touching the underlying feature space or retraining anything.
    penalty = 1 / weight, floored so a totally-rejected episode is heavily
    penalized but never produces a divide-by-zero.
    """
    penalty = np.array([1.0 / max(weights.get(eid, default_weight), 0.05)
                         for eid in neighbor_episode_ids])
    return distances * penalty


def weighted_knn_rerank(distances: np.ndarray, indices: np.ndarray,
                         neighbor_episode_ids: np.ndarray, feedback_db_path: str,
                         k: int) -> np.ndarray:
    """
    Drop-in re-ranking step: call this right after your existing
    `nbrs.kneighbors(query_vec)` call in recommender.py, before you pick the
    top-k neighbor(s) to build the recommendation.

        dist, idx = nbrs.kneighbors(query_vec, n_neighbors=k * 3)  # widen the pool
        keep_order = weighted_knn_rerank(dist[0], idx[0], episode_ids[idx[0]],
                                          "data/feedback.db", k=k)
        top_idx = idx[0][keep_order][:k]
    """
    weights = get_episode_weights(feedback_db_path)
    penalized = apply_feedback_penalty(distances, neighbor_episode_ids, weights)
    return np.argsort(penalized)


# --- Accept-rate trend, for the "show it improving" demo -------------------

def load_accept_rate_trend(feedback_db_path: str = "data/feedback.db",
                            window: int = 20) -> pd.DataFrame:
    con = sqlite3.connect(feedback_db_path)
    df = pd.read_sql("SELECT * FROM feedback ORDER BY timestamp", con)
    con.close()
    if df.empty:
        return df
    df["accepted"] = df["accepted"].astype(int)
    df["rolling_accept_rate"] = df["accepted"].rolling(window, min_periods=1).mean()
    return df


# --- Streamlit snippet for app.py (Accuracy Tracker panel) ------------------
#
# from src.feedback import load_accept_rate_trend   # or wherever you place this
#
# st.subheader("Feedback loop: accept-rate trend")
# trend_df = load_accept_rate_trend()
# if not trend_df.empty:
#     st.line_chart(trend_df.set_index("timestamp")["rolling_accept_rate"])
#     st.caption(f"Rolling {20}-recommendation acceptance rate. Rising trend = "
#                f"down-weighting rejected retrieval neighbors is working.")
# else:
#     st.info("No feedback logged yet — accept/reject a few recommendations to seed the chart.")