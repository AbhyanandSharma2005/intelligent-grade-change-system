"""Pydantic contracts for the Grade Change Intelligence API."""
from pydantic import BaseModel, Field


class Driver(BaseModel):
    feature: str
    impact: float


class PredictResponse(BaseModel):
    episode_id: int
    at_step: int
    risk: float = Field(ge=0, le=1)
    risk_level: str                      # "low" | "elevated" | "high"
    drivers: list[Driver]


class RecommendationOut(BaseModel):
    recommendation_id: str
    tag: str
    value: float
    rationale_tag: str                   # [Historical Data] etc.
    reason: str


class RecommendResponse(BaseModel):
    episode_id: int
    at_step: int
    risk: float
    action_required: bool
    neighbor_episodes: list[int]
    recommendations: list[RecommendationOut]


class CorrelationRow(BaseModel):
    tag: str
    avg_cross_corr: float
    flagged_by_shap: bool
    in_configured_loops: bool


class FeedbackIn(BaseModel):
    recommendation_id: str
    episode_id: int
    tag: str
    value: float
    rationale_tag: str
    accepted: bool


class FeedbackSummaryRow(BaseModel):
    rationale_tag: str
    n: int
    accept_rate: float


class EpisodeMeta(BaseModel):
    episode_id: int
    from_grade: str
    to_grade: str
    off_spec: bool
    time_to_stabilize_s: int

class OptimizedSetpoint(BaseModel):
    tag: str
    current: float
    optimized: float
    at_recipe_bound: bool


class OptimizeResponse(BaseModel):
    episode_id: int
    at_step: int
    risk_before: float
    risk_after: float
    setpoints: list[OptimizedSetpoint]


class ForecastPoint(BaseModel):
    horizon_s: int
    dev_pct_p10: float
    dev_pct_p50: float
    dev_pct_p90: float


class ForecastResponse(BaseModel):
    episode_id: int
    at_step: int
    spec_limit_pct: float
    points: list[ForecastPoint]
    breach_expected: bool          # P50 crosses the 2.5% limit
