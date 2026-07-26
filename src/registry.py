"""Minimal model registry: versioned artifacts with training metadata."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.config import MODEL_DIR

REGISTRY = Path(MODEL_DIR) / "registry"


def _data_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def register(metrics: dict, data_path: str) -> dict:
    """Snapshot the current deployed model as a new immutable version."""
    REGISTRY.mkdir(parents=True, exist_ok=True)
    existing = sorted(int(p.name[1:]) for p in REGISTRY.glob("v*"))
    version = (existing[-1] + 1) if existing else 1
    vdir = REGISTRY / f"v{version}"
    vdir.mkdir()

    shutil.copy(Path(MODEL_DIR) / "risk_model.joblib",
                vdir / "risk_model.joblib")
    meta = {"version": version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "training_data_sha256": _data_hash(data_path),
            "metrics": metrics}
    (vdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (REGISTRY / "latest.json").write_text(json.dumps(meta, indent=2))
    return meta


def latest() -> dict | None:
    p = REGISTRY / "latest.json"
    return json.loads(p.read_text()) if p.exists() else None
