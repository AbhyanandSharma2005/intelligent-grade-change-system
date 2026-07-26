"""
src/optimizer.py

Constrained setpoint optimizer: scipy.optimize (SLSQP) searches actuator
setpoints that minimize the calibrated risk model's predicted P(breach),
bounded by RECIPE_LIMITS and a ramp-rate cap. Complements — does not replace —
the k-NN recommender: /recommend returns the nearest historical fix,
/optimize returns what SLSQP finds by directly minimizing predicted risk.
Demo both side by side as "retrieval vs optimization".

Caveat worth stating to the client directly: the risk model only sees
engineered window features (rolling mean/slope over the trailing WINDOW
steps), not a process simulator. This treats predict_risk as a response
surface over each actuator's "_mean" feature, holding every other feature
(slopes, quality-tag stats, bw_dev_pct, bw_setpoint_delta) fixed at its
currently observed value — a reasonable local approximation of "what if this
actuator's trailing-window average had been different", not a dynamic
simulation of the ramp transient itself.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from src.config import ACTUATOR_TAGS, RECIPE_LIMITS
from src.model import predict_risk

RAMP_FRACTION = 0.12  # max single-step move, as a fraction of the actuator's
                       # full recipe range — the ramp-rate constraint
MAX_MOVES = 3          # cap actuators allowed to move so recs stay operator-digestible
BOUND_EPS = 1e-3       # tolerance for "at_recipe_bound" comparison


def _bounds_for(feats: dict):
    """Returns list of (feature_key, tag, lo, hi, recipe_lo, recipe_hi).
    lo/hi are the ramp-restricted search bounds; recipe_lo/recipe_hi are the
    hard RECIPE_LIMITS envelope, kept separately so we can flag when the
    optimizer's answer is pinned against the actual recipe limit (not just
    the tighter ramp bound)."""
    out = []
    for tag in ACTUATOR_TAGS:
        key = f"{tag}_mean"
        if key not in feats:
            continue
        current = float(feats[key])
        recipe_lo, recipe_hi = RECIPE_LIMITS[tag]
        ramp = (recipe_hi - recipe_lo) * RAMP_FRACTION
        lo, hi = max(recipe_lo, current - ramp), min(recipe_hi, current + ramp)
        if lo > hi:
            lo, hi = hi, lo
        out.append((key, tag, lo, hi, recipe_lo, recipe_hi))
    return out


def _risk_with_overrides(bundle, feats: dict, keys: list[str], x) -> float:
    trial = dict(feats)
    for k, v in zip(keys, x):
        trial[k] = float(v)
    return predict_risk(bundle, trial)


def optimize_setpoints(bundle, feats: dict, max_moves: int = MAX_MOVES) -> dict:
    """Matches api/schemas.py OptimizeResponse exactly:
    {risk_before, risk_after, setpoints: [{tag, current, optimized, at_recipe_bound}]}
    api/main.py unpacks this dict as **result into OptimizeResponse."""
    bound_rows = _bounds_for(feats)
    if not bound_rows:
        r = predict_risk(bundle, feats)
        return {"risk_before": r, "risk_after": r, "setpoints": []}

    keys = [b[0] for b in bound_rows]
    tags = [b[1] for b in bound_rows]
    scipy_bounds = [(b[2], b[3]) for b in bound_rows]
    recipe_bounds = {b[1]: (b[4], b[5]) for b in bound_rows}
    x0 = np.array([feats[k] for k in keys], dtype=float)
    risk_before = _risk_with_overrides(bundle, feats, keys, x0)

    result = minimize(lambda x: _risk_with_overrides(bundle, feats, keys, x),
                      x0, method="SLSQP", bounds=scipy_bounds,
                      options={"maxiter": 60, "ftol": 1e-6})
    x_final = result.x

    if max_moves is not None and max_moves < len(keys):
        deltas = np.abs(result.x - x0)
        keep_idx = np.argsort(-deltas)[:max_moves]
        fixed = x0.copy()
        free_bounds = [scipy_bounds[i] for i in keep_idx]

        def sparse_obj(xf):
            x_full = fixed.copy()
            x_full[keep_idx] = xf
            return _risk_with_overrides(bundle, feats, keys, x_full)

        r2 = minimize(sparse_obj, x0[keep_idx], method="SLSQP",
                      bounds=free_bounds, options={"maxiter": 60, "ftol": 1e-6})
        x_final = fixed.copy()
        x_final[keep_idx] = r2.x

    risk_after = _risk_with_overrides(bundle, feats, keys, x_final)

    setpoints = []
    for tag, cur, new in zip(tags, x0, x_final):
        recipe_lo, recipe_hi = recipe_bounds[tag]
        at_bound = (abs(new - recipe_lo) < BOUND_EPS
                   or abs(new - recipe_hi) < BOUND_EPS)
        setpoints.append({
            "tag": tag, "current": float(cur),
            "optimized": float(new), "at_recipe_bound": bool(at_bound),
        })

    return {
        "risk_before": float(risk_before),
        "risk_after": float(risk_after),
        "setpoints": setpoints,
    }