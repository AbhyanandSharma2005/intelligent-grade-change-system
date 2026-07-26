"""XGBoost deviation-risk classifier + SHAP explainability."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR


def train_risk_model(table: pd.DataFrame, model_dir: str = MODEL_DIR):
    y = table["label_off_spec"]
    X = table.drop(columns=["label_off_spec"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=0)
    model = xgb.XGBClassifier(n_estimators=250, max_depth=4,
                              learning_rate=0.08, subsample=0.9,
                              eval_metric="logloss")
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": list(X.columns), "auc": auc},
                Path(model_dir) / "risk_model.joblib")
    return model, auc


def load_risk_model(model_dir: str = MODEL_DIR):
    return joblib.load(Path(model_dir) / "risk_model.joblib")


def predict_risk(bundle, feats: dict) -> float:
    X = pd.DataFrame([feats])[bundle["columns"]]
    return float(bundle["model"].predict_proba(X)[0, 1])


def shap_top_drivers(bundle, feats: dict, top_n: int = 5):
    """Return [(feature, shap_value)] sorted by |impact| on current risk."""
    X = pd.DataFrame([feats])[bundle["columns"]]
    explainer = shap.TreeExplainer(bundle["model"])
    sv = explainer.shap_values(X)
    sv = np.asarray(sv).reshape(-1)
    order = np.argsort(-np.abs(sv))[:top_n]
    return [(bundle["columns"][i], float(sv[i])) for i in order]
