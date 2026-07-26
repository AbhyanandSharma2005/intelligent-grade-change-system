"""Replays the synthetic historian dataset as if it were a live feed."""
from typing import Iterator

import pandas as pd

from src.config import DATA_PATH, META_PATH
from src.ingestion.base import HistorianSource


class CSVReplaySource(HistorianSource):
    def __init__(self, data_path: str = DATA_PATH,
                 meta_path: str = META_PATH):
        self._episodes = pd.read_parquet(data_path)
        self._meta = pd.read_csv(meta_path)

    def list_episodes(self) -> pd.DataFrame:
        return self._meta.copy()

    def get_episode(self, episode_id: int) -> pd.DataFrame:
        ep = self._episodes[self._episodes["episode_id"] == episode_id]
        if ep.empty:
            raise KeyError(f"Unknown episode_id: {episode_id}")
        return ep.reset_index(drop=True)

    def stream(self, episode_id: int,
               start_step: int = 0) -> Iterator[dict]:
        ep = self.get_episode(episode_id)
        for _, row in ep.iloc[start_step:].iterrows():
            yield row.to_dict()
