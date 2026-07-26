"""Stub for a live OPC-UA connection to the mill DCS/QCS.

Deliberately unimplemented: it documents exactly what a production
deployment needs, and proves the architecture is integration-ready."""
from typing import Iterator

import pandas as pd

from src.ingestion.base import HistorianSource

# Example production tag mapping (Honeywell Experion / QCS namespace)
TAG_MAP = {
    "stock_flow": "ns=2;s=PM1.STOCKFLOW.PV",
    "filler_flow": "ns=2;s=PM1.FILLERFLOW.PV",
    "steam_pressure": "ns=2;s=PM1.DRYER.STEAM.PV",
    "machine_speed": "ns=2;s=PM1.SPEED.PV",
    "basis_weight": "ns=2;s=PM1.SCANNER.BW.PV",
    "moisture": "ns=2;s=PM1.SCANNER.MOIST.PV",
    "ash": "ns=2;s=PM1.SCANNER.ASH.PV",
    "caliper": "ns=2;s=PM1.SCANNER.CALIPER.PV",
    "dryer_humidity": "ns=2;s=PM1.DRYER.HUMIDITY.PV",
}


class OPCUASource(HistorianSource):
    """Would subscribe to the tags in TAG_MAP via asyncua and window
    incoming samples into episodes keyed on the QCS grade-change trigger."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def list_episodes(self) -> pd.DataFrame:
        raise NotImplementedError("Requires live plant connection")

    def get_episode(self, episode_id: int) -> pd.DataFrame:
        raise NotImplementedError("Requires live plant connection")

    def stream(self, episode_id: int, start_step: int = 0) -> Iterator[dict]:
        raise NotImplementedError("Requires live plant connection")
