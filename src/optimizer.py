"""Constrained setpoint optimizer.

Searches actuator setpoints that minimize predicted deviation risk,
subject to (a) recipe/actuator hard limits and (b) a ramp-rate constraint
(max move per advisory). Uses differential evolution because tree-model
risk surfaces are piecewise-constant (gradient methods stall)."""
import numpy as np
from scipy.optimize import differential_evolution

from src.config import ACTUATOR_TAGS, RECIPE_LIMITS
from src.model import predict_risk

MAX_MOVE_FRAC = 0.06      # max move per advisory: 6% of recipe range
RAMP_STEPS = 12           # assumed ramp length for the slope feature (~1 min)
BOUND_TOL = 1e-6


def optimize_setpoints(bundle, live_feats: dict) -> dict:
    """Returns current vs optimized setpoints and risk before/after."""
    current = {a: float(live_feats[f"{a}_mean"]) for a in ACTUATOR_TAGS}

    bounds = []
    for a in ACTUATOR_TAGS:
        lo, hi = RECIPE_LIMITS[a]
        max_move = MAX_MOVE_FRAC * (hi - lo)
        bounds.append((max(lo, current[a] - max_move),
                       min(hi, current[a] + max_move)))

    def risk_of(x: np.ndarray) -> float:
        f = dict(live_feats)
        for a, v in zip(ACTUATOR_TAGS, x):
            f[f"{a}_mean"] = float(v)
            f[f"{a}_slope"] = (float(v) - current[a]) / RAMP_STEPS
        return predict_risk(bundle, f)

    risk_before = risk_of(np.array([current[a] for a in ACTUATOR_TAGS]))
    result = differential_evolution(risk_of, bounds, maxiter=25, popsize=12,
                                    tol=1e-3, seed=0, polish=False)

    setpoints = []
    for a, v, (lo, hi) in zip(ACTUATOR_TAGS, result.x, bounds):
        r_lo, r_hi = RECIPE_LIMITS[a]
        setpoints.append({
            "tag": a,
            "current": round(current[a], 1),
            "optimized": round(float(v), 1),
            "at_recipe_bound": (abs(v - r_lo) < BOUND_TOL
                                or abs(v - r_hi) < BOUND_TOL),
        })
    return {"risk_before": round(risk_before, 3),
            "risk_after": round(float(result.fun), 3),
            "setpoints": setpoints}
