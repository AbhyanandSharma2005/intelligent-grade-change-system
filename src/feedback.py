"""Operator feedback loop: SQLite log + accept-rate aggregation."""
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import FEEDBACK_DB


def _conn():
    Path(FEEDBACK_DB).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        recommendation_id TEXT PRIMARY KEY,
        episode_id INTEGER, tag TEXT, value REAL,
        rationale_tag TEXT, accepted INTEGER, ts TEXT)""")
    return conn


def new_recommendation_id() -> str:
    return uuid.uuid4().hex[:10]


def log_feedback(rec_id: str, episode_id: int, tag: str, value: float,
                 rationale_tag: str, accepted: bool) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?,?,?)",
                  (rec_id, episode_id, tag, value, rationale_tag,
                   int(accepted), datetime.now(timezone.utc).isoformat()))


def accuracy_summary() -> pd.DataFrame:
    with _conn() as c:
        df = pd.read_sql("SELECT * FROM feedback", c)
    if df.empty:
        return pd.DataFrame(columns=["rationale_tag", "n", "accept_rate"])
    g = df.groupby("rationale_tag").agg(
        n=("accepted", "size"), accept_rate=("accepted", "mean")).reset_index()
    g["accept_rate"] = (g["accept_rate"] * 100).round(1)
    return g


def feedback_log() -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql("SELECT * FROM feedback ORDER BY ts DESC", c)


def accept_rate_trend(window: int = 20) -> pd.DataFrame:
    """Rolling accept-rate over time, ordered oldest -> newest — the 'feedback
    loop actually learns' chart. Shows whether down-weighting rejected
    retrieval neighbors (see src/recommender.py) is improving acceptance
    over time. Returns an empty DataFrame if nothing's been logged yet."""
    df = feedback_log()
    if df.empty:
        return df
    df = df.sort_values("ts").reset_index(drop=True)
    df["accepted"] = df["accepted"].astype(int)
    df["rolling_accept_rate"] = df["accepted"].rolling(window, min_periods=1).mean()
    return df