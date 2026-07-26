# Grade Change Intelligence — Paper Making MVP

A predictive + explainable co-pilot for MD grade transitions. Sits on top of an existing QCS: predicts basis-weight deviation risk (>±2.5% of setpoint) during a live grade change, recommends corrective setpoints with a tagged rationale, discovers correlations not in the configured control loops, and logs operator accept/reject feedback.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           GRADE CHANGE INTELLIGENCE SYSTEM                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                              ▲
                                              │  Basis weight scans (historian)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                           │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │  Episode Parquet │    │  Episode Meta    │    │  Feature Table (Parquet) │   │
│  │  (400 episodes)  │    │  (grades, tags)  │    │  (precomputed features)  │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                              ▲
                                              │  Feature extraction @ step T
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL LAYER                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                    XGBoost Risk Model (Calibrated)                        │    │
│  │  Input: 40+ engineered features  →  Output: P(breach ±2.5%) ∈ [0, 1]     │    │
│  │  Calibration: IsotonicRegression  │  Explainability: Native SHAP (TreeSHAP)│    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                    XGBoost Forecast Model (Quantile Regression)           │    │
│  │  Input: rolling window features  →  Output: P10/P50/P90 deviation @ horizons│    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                              ▲
                                              │  Risk + SHAP drivers
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │  /health    │ │ /predict    │ │ /recommend  │ │ /optimize   │ │ /forecast │  │
│  │  /episodes  │ │ /correlations│ │ /feedback   │ │ /ws/live    │ │ /model/info│  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
│  • Pydantic contracts  • WebSocket live feed  • Operator feedback logging       │
│  • Model registry (MLflow-style)  • Constrained optimizer (SLSQP)                │
└─────────────────────────────────────────────────────────────────────────────────┘
                          ▲                       ▲
                          │ REST / WS             │ REST
                          ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            DASHBOARD (Streamlit)                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  LIVE TRANSITION REPLAY                                                  │   │
│  │  Episode selector ▸ Time slider ▸ Basis weight vs setpoint chart        │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────────────────────────────┐   │
│  │ RISK SCORE   │ │ DEVIATION    │ │ TOP SHAP DRIVERS                      │   │
│  │ P(breach) %  │ │ FORECAST     │ │ (tag, signed impact)                  │   │
│  │ Gauge + bar  │ │ P10/P50/P90  │ │ Auto-discovered correlations panel   │   │
│  └──────────────┘ └──────────────┘ └───────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  RECOMMENDATIONS (when risk > 35%)                                       │   │
│  │  [Tag → Value]  Rationale tag  [Accept] [Reject]  →  Feedback log (SQLite)│   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  ACCURACY TRACKER:  Accept rate by rationale tag + full feedback table    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start (Local Development)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic historian data (400 grade-change episodes)
python scripts/generate_data.py

# 3. Train risk model + artifacts
python scripts/train.py

# 4. (Optional) Train forecast model
python scripts/train_forecast.py

# 5. Launch API server
python -m uvicorn api.main:app --reload --port 8080

# 6. Launch dashboard (in another terminal)
streamlit run app.py
```

**Access:**
- API: http://localhost:8080/docs
- Dashboard: http://localhost:8501

---

## Docker Deployment (Production)

### Prerequisites
- Docker 24+
- Docker Compose v2+

### One-command deployment

```bash
# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Services

| Service | Port | Health Check |
|---------|------|--------------|
| **api** | 8080 | `GET /health` |
| **dashboard** | 8501 | `GET /_stcore/health` |

### Data Persistence

Volumes are mounted for:
- `./data` → `/app/data` (historian parquet + metadata)
- `./models` → `/app/models` (trained model artifacts + registry)
- `./data/feedback.db` → `/app/data/feedback.db` (operator feedback SQLite)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/app/models` | Path to model artifacts |
| `DATA_PATH` | `/app/data/episodes.parquet` | Episode parquet file |
| `META_PATH` | `/app/data/episodes_meta.csv` | Episode metadata CSV |
| `DEVIATION_LIMIT_PCT` | `2.5` | Basis weight deviation limit (%) |
| `RISK_ACTION_THRESHOLD` | `0.35` | Risk threshold for recommendations |

Override via `.env` file or `docker compose -f docker-compose.yml -f docker-compose.override.yml up`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + model AUC |
| `GET` | `/model/info` | Registered model metadata |
| `GET` | `/episodes` | List all grade-change episodes |
| `GET` | `/predict/{episode_id}` | Risk prediction at step |
| `GET` | `/recommend/{episode_id}` | Setpoint recommendations |
| `GET` | `/optimize/{episode_id}` | Constrained optimizer (SLSQP) |
| `GET` | `/forecast/{episode_id}` | Quantile deviation forecast |
| `GET` | `/correlations` | Discovered cross-correlations |
| `WS` | `/ws/live/{episode_id}` | Live scanner feed (risk + BW) |
| `POST` | `/feedback` | Log operator accept/reject |
| `GET` | `/feedback/summary` | Acceptance rate by rationale tag |

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push:

1. **Lint & Type Check** — `ruff`, `mypy`
2. **Unit Tests** — `pytest` (API contracts, optimizer, forecast)
3. **Docker Build** — Multi-stage builds for API & Dashboard
4. **Smoke Test** — `docker compose up` + health checks on `main` branch

```yaml
# Trigger on push to main or PR
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```

---

## Project Structure

```
intelligent-grade-change-system/
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── api/
│   ├── main.py                 # FastAPI app + routes
│   ├── schemas.py              # Pydantic contracts
│   └── deps.py                 # AppState dependency injection
├── src/
│   ├── config.py               # Constants, tag lists
│   ├── model.py                # Risk model load/predict/SHAP
│   ├── forecast.py             # Quantile forecast model
│   ├── optimizer.py            # Constrained SLSQP optimizer
│   ├── recommender.py          # KNN-based setpoint recommender
│   ├── features.py             # Episode feature engineering
│   ├── correlations.py         # Cross-correlation discovery
│   ├── feedback.py             # SQLite feedback logging
│   ├── registry.py             # Model registry (MLflow-style)
│   ├── training.py             # XGBoost training pipeline
│   ├── backtest.py             # Rolling-origin backtest
│   ├── drift.py                # Population stability index
│   └── ingestion/              # Historian adapters (CSV replay, OPC-UA stub)
├── scripts/
│   ├── generate_data.py        # Synthetic data generator
│   ├── train.py                # Train risk model
│   ├── train_forecast.py       # Train forecast model
│   ├── train_v2.py             # Calibrated model training
│   ├── backtest.py             # Backtest runner
│   └── smoke_test.py           # Quick integration test
├── tests/
│   ├── test_api.py             # API contract tests
│   └── test_optimizer_forecast.py
├── models/                     # Trained artifacts (git-tracked)
│   ├── risk_model.joblib
│   ├── forecast.joblib
│   ├── feature_table.parquet
│   └── registry/
├── data/                       # Historian data (git-tracked samples)
│   ├── episodes.parquet
│   ├── episodes_meta.csv
│   └── feedback.db
├── app.py                      # Streamlit dashboard
├── Dockerfile.api              # API multi-stage build
├── Dockerfile.dashboard        # Dashboard multi-stage build
├── docker-compose.yml          # Service orchestration
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Model Training Details

### Risk Model
- **Algorithm**: XGBoost (binary classification) + Isotonic calibration
- **Target**: Basis weight breach > ±2.5% of setpoint within prediction horizon
- **Features**: 40+ engineered (rolling stats, cross-correlations, grade-pair encoding, actuator positions)
- **Explainability**: Native TreeSHAP (no KernelSHAP approximation)
- **Validation**: Rolling-origin backtest (80/20 temporal split)

### Forecast Model
- **Algorithm**: XGBoost quantile regression (P10/P50/P90)
- **Horizons**: 30s, 60s, 120s, 300s ahead
- **Features**: Rolling window statistics of basis weight & actuators

---

## Feedback Loop

Operators click **Accept** / **Reject** on each recommendation → logged to SQLite with:
- `recommendation_id`, `episode_id`, `tag`, `value`, `rationale_tag`, `accepted`, `timestamp`

Dashboard **Accuracy Tracker** shows:
- Acceptance rate by rationale tag
- Full feedback log with filtering

---

## Extending

### New Historian Source
Implement `src.ingestion.base.HistorianSource`:
```python
class MyHistorian(HistorianSource):
    def list_episodes(self) -> pd.DataFrame: ...
    def get_episode(self, eid: int) -> pd.DataFrame: ...
```

Register in `api/deps.py` → `build_state()`.

### New Control Loop Tag
Add to `ACTUATOR_TAGS` in `src/config.py` → retrain → redeploy.

---

## License

MIT — Internal use for paper-making grade-change optimization.