"""Singleton application state, loaded once at startup."""
from dataclasses import dataclass

import pandas as pd

from src.config import MODEL_DIR
from src.correlations import discover_new_correlations
from src.ingestion.csv_replay import CSVReplaySource
from src.model import load_risk_model
from src.recommender import SetpointRecommender
from src.forecast import load_forecast


@dataclass
class AppState:
    source: CSVReplaySource
    bundle: dict
    recommender: SetpointRecommender
    correlations: pd.DataFrame
    forecast_bundle: dict | None

def build_state() -> AppState:
    source = CSVReplaySource()
    bundle = load_risk_model(MODEL_DIR)
    table = pd.read_parquet(f"{MODEL_DIR}/feature_table.parquet")
    meta = source.list_episodes()
    episodes = pd.concat(
        [source.get_episode(e) for e in meta["episode_id"]],
        ignore_index=True)
    recommender = SetpointRecommender(table, meta, episodes)
    # Correlation discovery is expensive -> computed once at startup
    correlations = discover_new_correlations(episodes, shap_drivers=[])
    forecast_bundle=load_forecast()
    return AppState(source=source, bundle=bundle,
                    recommender=recommender, correlations=correlations,forecast_bundle=forecast_bundle)
