import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.features import episode_features
from src.model import load_risk_model, predict_risk, shap_top_drivers

episodes = pd.read_parquet("data/episodes.parquet")
meta = pd.read_csv("data/episodes_meta.csv")

bad_id = meta[meta.off_spec].episode_id.iloc[0]
good_id = meta[~meta.off_spec].episode_id.iloc[0]
bundle = load_risk_model()

for label, eid in [("OFF-SPEC", bad_id), ("IN-SPEC", good_id)]:
    ep = episodes[episodes.episode_id == eid].reset_index(drop=True)
    f = episode_features(ep, at_step=200)
    print(f"{label} episode {eid} -> risk: {predict_risk(bundle, f):.2f}")
    print("  top drivers:", shap_top_drivers(bundle, f, top_n=3))
