"""Central configuration: tags, recipe limits, configured control loops."""

SAMPLE_SECONDS = 5
PRE_TRIGGER_MIN = 10          # minutes of data before grade-change trigger
POST_TRIGGER_MIN = 20         # minutes after trigger
STEPS_PER_EPISODE = (PRE_TRIGGER_MIN + POST_TRIGGER_MIN) * 60 // SAMPLE_SECONDS
TRIGGER_STEP = PRE_TRIGGER_MIN * 60 // SAMPLE_SECONDS

DEVIATION_LIMIT_PCT = 2.5     # basis weight spec: ±2.5% of setpoint
STABLE_WINDOW_STEPS = 24      # 2 min inside band = "stabilized"

# Tags visible to the system (dryer_humidity is measured but NOT in any loop)
PROCESS_TAGS = ["stock_flow", "filler_flow", "steam_pressure",
                "machine_speed", "moisture", "ash", "caliper",
                "dryer_humidity"]
QUALITY_TAG = "basis_weight"
ACTUATOR_TAGS = ["stock_flow", "filler_flow", "steam_pressure", "machine_speed"]

# Relationships already encoded in the QCS control loops (the "known" set).
# Anything else the engine finds influencing basis weight is NEW.
CONFIGURED_LOOPS = {
    "basis_weight": ["stock_flow", "machine_speed"],
    "moisture": ["steam_pressure"],
    "ash": ["filler_flow"],
}

# Recipe / actuator hard limits (min, max) — recommendations are clipped here
RECIPE_LIMITS = {
    "stock_flow": (4000.0, 9000.0),      # L/min
    "filler_flow": (300.0, 900.0),       # L/min
    "steam_pressure": (250.0, 480.0),    # kPa
    "machine_speed": (800.0, 1300.0),    # m/min
}

# Grade recipes: basis-weight setpoint + nominal actuator targets
GRADES = {
    "GSM_60":  {"bw_setpoint": 60.0,  "stock_flow": 5200, "filler_flow": 420,
                "steam_pressure": 320, "machine_speed": 1200},
    "GSM_80":  {"bw_setpoint": 80.0,  "stock_flow": 6400, "filler_flow": 610,
                "steam_pressure": 360, "machine_speed": 1050},
    "GSM_100": {"bw_setpoint": 100.0, "stock_flow": 7600, "filler_flow": 780,
                "steam_pressure": 400, "machine_speed": 900},
}

DATA_PATH = "data/episodes.parquet"
META_PATH = "data/episodes_meta.csv"
MODEL_DIR = "models"
FEEDBACK_DB = "data/feedback.db"
