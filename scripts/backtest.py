"""Backtest: replay held-out episodes and measure alarm quality."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import (DATA_PATH, DEVIATION_LIMIT_PCT, META_PATH,
                        SAMPLE_SECONDS, TRIGGER_STEP)
from src.features import episode_features
from src.model import load_risk_model, predict_risk

ALARM_THRESHOLD = 0.5
STRIDE = 4


def breach_step(ep: pd.DataFrame) -> int | None:
    dev = (ep["basis_weight"] - ep["bw_setpoint"]).abs() \
          / ep["bw_setpoint"] * 100
    post = dev.iloc[TRIGGER_STEP:]
    hits = post[post > DEVIATION_LIMIT_PCT]
    return int(hits.index[0]) if len(hits) else None


def main():
    bundle = load_risk_model()
    episodes = pd.read_parquet(DATA_PATH)
    meta = pd.read_csv(META_PATH).set_index("episode_id")
    test_ids = bundle["test_episodes"]

    rows = []
    for eid in test_ids:
        ep = episodes[episodes.episode_id == eid].reset_index(drop=True)
        alarm_step = None
        for t in range(TRIGGER_STEP + 20, len(ep), STRIDE):
            if predict_risk(bundle, episode_features(ep, at_step=t)) \
                    >= ALARM_THRESHOLD:
                alarm_step = t
                break
        b = breach_step(ep)
        lead = ((b - alarm_step) * SAMPLE_SECONDS
                if (b is not None and alarm_step is not None
                    and alarm_step < b) else None)
        rows.append({"episode_id": eid,
                     "off_spec": bool(meta.loc[eid, "off_spec"]),
                     "alarmed": alarm_step is not None,
                     "alarm_step": alarm_step, "breach_step": b,
                     "lead_time_s": lead})

    df = pd.DataFrame(rows)
    df.to_csv("backtest_report.csv", index=False)

    pos, neg = df[df.off_spec], df[~df.off_spec]
    leads = df["lead_time_s"].dropna()
    print(f"Held-out episodes:      {len(df)}")
    print(f"Detection rate:         {pos['alarmed'].mean():.1%} "
          f"({pos['alarmed'].sum()}/{len(pos)} off-spec caught)")
    print(f"False alarm rate:       {neg['alarmed'].mean():.1%} "
          f"({neg['alarmed'].sum()}/{len(neg)} in-spec flagged)")
    if len(leads):
        print(f"Median alarm lead time: {np.median(leads):.0f} s "
              f"(p25={np.percentile(leads, 25):.0f}s, "
              f"p75={np.percentile(leads, 75):.0f}s)")
    print("Per-episode detail -> backtest_report.csv")


if __name__ == "__main__":
    main()
