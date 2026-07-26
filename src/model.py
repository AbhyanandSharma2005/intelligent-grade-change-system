"""Deviation-risk model: calibrated XGBoost + native SHAP explainability."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR


def _precision_at_recall(y_true, proba, min_recall=0.8) -> float:
    prec, rec, _ = precision_recall_curve(y_true, proba)
    valid = prec[rec >= min_recall]
    return float(valid.max()) if len(valid) else 0.0


def train_risk_model(table: pd.DataFrame, model_dir: str = MODEL_DIR):
    """Episode-level split (one row per episode => no window leakage),
    isotonic calibration, and domain metrics."""
    y = table["label_off_spec"]
    X = table.drop(columns=["label_off_spec"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=0)

    # Raw model: kept for SHAP explanations
    raw = xgb.XGBClassifier(n_estimators=250, max_depth=4,
                            learning_rate=0.08, subsample=0.9,
                            eval_metric="logloss")
    raw.fit(Xtr, ytr)

    # Calibrated model: what production risk scores come from
    calibrated = CalibratedClassifierCV(
        xgb.XGBClassifier(n_estimators=250, max_depth=4,
                          learning_rate=0.08, subsample=0.9,
                          eval_metric="logloss"),
        method="isotonic", cv=3)
    calibrated.fit(Xtr, ytr)

    proba = calibrated.predict_proba(Xte)[:, 1]
    metrics = {
        "auc": float(roc_auc_score(yte, proba)),
        "brier": float(brier_score_loss(yte, proba)),
        "precision_at_recall80": _precision_at_recall(yte, proba),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "positive_rate": float(y.mean()),
    }

    bundle = {
        "model": calibrated,
        "explainer_model": raw,
        "columns": list(X.columns),
        "metrics": metrics,
        "auc": metrics["auc"],                    # backward compat
        "test_episodes": [int(i) for i in Xte.index],
        # Reference feature distribution quantiles for drift detection
        "feature_ref_quantiles": Xtr.quantile(
            np.linspace(0, 1, 11)).to_dict("list"),
    }
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, Path(model_dir) / "risk_model.joblib")
    return bundle, metrics


def load_risk_model(model_dir: str = MODEL_DIR):
    return joblib.load(Path(model_dir) / "risk_model.joblib")


def predict_risk(bundle, feats: dict) -> float:
    X = pd.DataFrame([feats])[bundle["columns"]]
    return float(bundle["model"].predict_proba(X)[0, 1])


def shap_top_drivers(bundle, feats: dict, top_n: int = 5):
    """Native XGBoost SHAP (pred_contribs) on the raw model."""
    X = pd.DataFrame([feats])[bundle["columns"]]
    booster = bundle["explainer_model"].get_booster()
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    order = np.argsort(-np.abs(contribs))[:top_n]
    return [(bundle["columns"][i], float(contribs[i])) for i in order]
