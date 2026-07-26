"""Historian source abstraction.

In production this is the integration seam: the API layer only ever talks
to a HistorianSource, so swapping CSV replay for a live OPC-UA / PI / PHD
connection requires zero changes upstream."""
from abc import ABC, abstractmethod
from typing import Iterator

import pandas as pd


class HistorianSource(ABC):
    """Contract for any source of grade-change process data."""

    @abstractmethod
    def list_episodes(self) -> pd.DataFrame:
        """Return episode metadata (episode_id, grades, labels...)."""

    @abstractmethod
    def get_episode(self, episode_id: int) -> pd.DataFrame:
        """Return the full time series for one grade-change episode."""

    @abstractmethod
    def stream(self, episode_id: int,
               start_step: int = 0) -> Iterator[dict]:
        """Yield samples one at a time, as a live scanner feed would."""
