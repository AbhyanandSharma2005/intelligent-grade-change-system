"""Train the deviation-risk model and cache the feature table."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import DATA_PATH, META_PATH, MODEL_DIR
from src.features import build_training_table
from src.model import train_risk_model


def main():
    Path(MODEL_DIR).mkdir(exist_ok=True)   # the fix flagged earlier
    episodes = pd.read_parquet(DATA_PATH)
    meta = pd.read_csv(META_PATH)
    table = build_training_table(episodes, meta)
    table.to_parquet(f"{MODEL_DIR}/feature_table.parquet")
    _, auc = train_risk_model(table)
    print(f"Risk model trained. Hold-out AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
