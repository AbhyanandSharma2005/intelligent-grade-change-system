"""Generate the synthetic historian dataset."""
import argparse
from pathlib import Path

from src.data_generator import generate_dataset
from src.config import DATA_PATH, META_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--failure-rate", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    episodes, meta = generate_dataset(args.episodes, args.failure_rate,
                                      args.seed)
    Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(DATA_PATH, index=False)
    meta.to_csv(META_PATH, index=False)
    print(f"{args.episodes} episodes -> {DATA_PATH}")
    print(f"Off-spec rate: {meta['off_spec'].mean():.1%}")


if __name__ == "__main__":
    main()
