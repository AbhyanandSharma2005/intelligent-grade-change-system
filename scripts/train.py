"""Train, evaluate, and register the deviation-risk model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import DATA_PATH, META_PATH, MODEL_DIR
from src.features import build_training_table
from src.model import train_risk_model
from src.registry import register


def main():
    Path(MODEL_DIR).mkdir(exist_ok=True)
    episodes = pd.read_parquet(DATA_PATH)
    meta = pd.read_csv(META_PATH)
    table = build_training_table(episodes, meta)
    table.to_parquet(f"{MODEL_DIR}/feature_table.parquet")

    _, metrics = train_risk_model(table)
    version_meta = register(metrics, DATA_PATH)

    print(f"Registered model v{version_meta['version']}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
