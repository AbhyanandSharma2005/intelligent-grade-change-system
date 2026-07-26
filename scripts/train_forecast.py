"""Train the quantile trajectory forecaster (separate from risk model)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import DATA_PATH
from src.forecast import train_forecast


def main():
    episodes = pd.read_parquet(DATA_PATH)
    train_forecast(episodes)
    print("Forecast models (3 horizons x 3 quantiles) -> models/forecast.joblib")


if __name__ == "__main__":
    main()
