"""Grade Change Intelligence API.

Run:  python -m uvicorn api.main:app --reload
Docs: http://localhost:8080/docs
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from api import schemas
from api.deps import AppState, build_state
from src.feedback import accuracy_summary, log_feedback, new_recommendation_id
from src.features import episode_features
from src.model import predict_risk, shap_top_drivers
from src.registry import latest as registry_latest
from src.config import DEVIATION_LIMIT_PCT
from src.forecast import forecast_deviation
from src.optimizer import optimize_setpoints


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("gci.api")

RISK_ACTION_THRESHOLD = 0.35
state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    log.info("Loading model artifacts and historian source...")
    state = build_state()
    log.info("Ready. Model AUC=%.3f, %d episodes indexed.",
             state.bundle["auc"], len(state.source.list_episodes()))
    yield


app = FastAPI(title="Grade Change Intelligence API",
              version="1.1.0", lifespan=lifespan)


def _risk_level(risk: float) -> str:
    return "high" if risk > 0.5 else "elevated" if risk > 0.35 else "low"


def _predict(episode_id: int, at_step: int):
    try:
        ep = state.source.get_episode(episode_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if not 40 <= at_step < len(ep):
        raise HTTPException(422, f"at_step must be in [40, {len(ep) - 1}]")
    feats = episode_features(ep, at_step=at_step)
    risk = predict_risk(state.bundle, feats)
    return ep, feats, risk


@app.get("/health")
def health():
    return {"status": "ok", "model_auc": round(state.bundle["auc"], 3)}


@app.get("/model/info")
def model_info():
    """Metadata for the currently deployed model version (from registry)."""
    meta = registry_latest()
    if meta is None:
        raise HTTPException(404, "No registered model")
    return meta


@app.get("/episodes", response_model=list[schemas.EpisodeMeta])
def episodes():
    cols = ["episode_id", "from_grade", "to_grade",
            "off_spec", "time_to_stabilize_s"]
    return state.source.list_episodes()[cols].to_dict("records")


@app.get("/predict/{episode_id}", response_model=schemas.PredictResponse)
def predict(episode_id: int, at_step: int = 200):
    _, feats, risk = _predict(episode_id, at_step)
    drivers = [schemas.Driver(feature=f, impact=v)
               for f, v in shap_top_drivers(state.bundle, feats)]
    return schemas.PredictResponse(episode_id=episode_id, at_step=at_step,
                                   risk=risk, risk_level=_risk_level(risk),
                                   drivers=drivers)


@app.get("/recommend/{episode_id}", response_model=schemas.RecommendResponse)
def recommend(episode_id: int, at_step: int = 200):
    _, feats, risk = _predict(episode_id, at_step)
    action = risk > RISK_ACTION_THRESHOLD
    recs, neighbors = [], []
    if action:
        new_tags = state.correlations["tag"].tolist() \
            if not state.correlations.empty else []
        raw, neighbors = state.recommender.recommend(feats, new_tags)
        recs = [schemas.RecommendationOut(
            recommendation_id=new_recommendation_id(),
            tag=r.tag, value=r.value,
            rationale_tag=r.primary_tag, reason=r.reason) for r in raw]
    return schemas.RecommendResponse(
        episode_id=episode_id, at_step=at_step, risk=risk,
        action_required=action, neighbor_episodes=list(neighbors),
        recommendations=recs)


@app.get("/correlations", response_model=list[schemas.CorrelationRow])
def correlations():
    return state.correlations.to_dict("records")


@app.post("/feedback", status_code=201)
def feedback(fb: schemas.FeedbackIn):
    log_feedback(fb.recommendation_id, fb.episode_id, fb.tag,
                 fb.value, fb.rationale_tag, fb.accepted)
    log.info("Feedback: %s accepted=%s", fb.recommendation_id, fb.accepted)
    return {"logged": True}


@app.get("/feedback/summary",
         response_model=list[schemas.FeedbackSummaryRow])
def feedback_summary():
    return accuracy_summary().to_dict("records")


@app.websocket("/ws/live/{episode_id}")
async def live_feed(ws: WebSocket, episode_id: int):
    """Streams the episode as a live scanner feed with rolling risk."""
    await ws.accept()
    try:
        ep = state.source.get_episode(episode_id)
        for t in range(60, len(ep), 4):          # every 4th sample
            feats = episode_features(ep, at_step=t)
            risk = predict_risk(state.bundle, feats)
            row = ep.iloc[t]
            await ws.send_json({
                "step": t,
                "basis_weight": round(float(row["basis_weight"]), 2),
                "bw_setpoint": round(float(row["bw_setpoint"]), 2),
                "risk": round(risk, 3),
                "risk_level": _risk_level(risk)})
            await asyncio.sleep(0.25)            # replay pacing
        await ws.close()
    except WebSocketDisconnect:
        log.info("Live feed for episode %d disconnected", episode_id)
    except KeyError:
        await ws.send_json({"error": f"unknown episode {episode_id}"})
        await ws.close()

@app.get("/optimize/{episode_id}", response_model=schemas.OptimizeResponse)
def optimize(episode_id: int, at_step: int = 200):
    """Constrained optimizer: setpoints minimizing predicted risk within
    recipe limits and ramp-rate constraints."""
    _, feats, _ = _predict(episode_id, at_step)
    result = optimize_setpoints(state.bundle, feats)
    return schemas.OptimizeResponse(episode_id=episode_id, at_step=at_step,
                                    **result)


@app.get("/forecast/{episode_id}", response_model=schemas.ForecastResponse)
def forecast(episode_id: int, at_step: int = 200):
    """Future basis-weight deviation if the current trend continues."""
    if state.forecast_bundle is None:
        raise HTTPException(503, "Forecast model not trained "
                                 "(run scripts/train_forecast.py)")
    _, feats, _ = _predict(episode_id, at_step)
    points = forecast_deviation(state.forecast_bundle, feats)
    breach = any(p["dev_pct_p50"] > DEVIATION_LIMIT_PCT for p in points)
    return schemas.ForecastResponse(
        episode_id=episode_id, at_step=at_step,
        spec_limit_pct=DEVIATION_LIMIT_PCT,
        points=points, breach_expected=breach)
