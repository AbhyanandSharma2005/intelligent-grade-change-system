"""Grade Change Intelligence — operator dashboard."""
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (ACTUATOR_TAGS, DATA_PATH, DEVIATION_LIMIT_PCT,
                        META_PATH, MODEL_DIR, QUALITY_TAG, TRIGGER_STEP)
from src.correlations import discover_new_correlations
from src.features import episode_features
from src.feedback import (accept_rate_trend, accuracy_summary, feedback_log,
                          log_feedback, new_recommendation_id)
from src.forecast import forecast_deviation, load_forecast
from src.model import load_risk_model, predict_risk, shap_top_drivers
from src.optimizer import optimize_setpoints
from src.recommender import SetpointRecommender

st.set_page_config(page_title="Grade Change Intelligence", layout="wide")


@st.cache_resource
def load_all():
    episodes = pd.read_parquet(DATA_PATH)
    meta = pd.read_csv(META_PATH)
    bundle = load_risk_model(MODEL_DIR)
    table = pd.read_parquet(f"{MODEL_DIR}/feature_table.parquet")
    rec = SetpointRecommender(table, meta, episodes)
    forecast_bundle = load_forecast(MODEL_DIR)  # None if not trained yet
    return episodes, meta, bundle, rec, forecast_bundle


episodes, meta, bundle, recommender, forecast_bundle = load_all()

st.title("Grade Change Intelligence — MD Transition Co-pilot")
st.caption("Predictive + explainable layer on top of the QCS grade change program")

# ── Sidebar: pick a 'live' episode to replay ─────────────────────────
with st.sidebar:
    st.header("Live transition")
    eid = st.selectbox("Replay episode as live feed",
                       meta["episode_id"].tolist())
    row = meta.set_index("episode_id").loc[eid]
    st.write(f"**{row['from_grade']} → {row['to_grade']}**")
    ep = episodes[episodes["episode_id"] == eid].reset_index(drop=True)
    t_now = st.slider("Current time step", TRIGGER_STEP + 40, len(ep) - 1,
                      TRIGGER_STEP + 60)

# ── Live prediction ──────────────────────────────────────────────────
feats = episode_features(ep, at_step=t_now)
risk = predict_risk(bundle, feats)
drivers = shap_top_drivers(bundle, feats)
new_corr = discover_new_correlations(episodes, drivers)
new_corr_tags = new_corr["tag"].tolist() if not new_corr.empty else []

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.subheader("Basis weight vs setpoint")
    view = ep.iloc[:t_now]
    band = ep["bw_setpoint"] * DEVIATION_LIMIT_PCT / 100
    chart = pd.DataFrame({
        "basis_weight": view[QUALITY_TAG],
        "setpoint": view["bw_setpoint"],
        "upper_limit": (ep["bw_setpoint"] + band).iloc[:t_now],
        "lower_limit": (ep["bw_setpoint"] - band).iloc[:t_now]})
    st.line_chart(chart)
with c2:
    st.subheader("Deviation risk")
    st.metric("P(breach ±2.5%)", f"{risk:.0%}",
              delta="HIGH RISK" if risk > 0.5 else "in control",
              delta_color="inverse" if risk > 0.5 else "normal")
    st.progress(min(risk, 1.0))
with c3:
    st.subheader("Top risk drivers (SHAP)")
    for f, v in drivers:
        st.write(f"- `{f}` ({'+' if v > 0 else ''}{v:.3f})")

# ── Deviation forecast (quantile bands) ─────────────────────────────
st.divider()
st.subheader("Deviation forecast — if the current trend continues")
if forecast_bundle is None:
    st.info("Forecast model not trained yet — run `python scripts/train_forecast.py`.")
else:
    points = forecast_deviation(forecast_bundle, feats)
    df_fc = pd.DataFrame(points).set_index("horizon_s")
    st.line_chart(df_fc[["dev_pct_p10", "dev_pct_p50", "dev_pct_p90"]])
    st.caption(f"P10 / P50 / P90 basis-weight deviation (%) at each horizon. "
              f"Spec limit is ±{DEVIATION_LIMIT_PCT}%.")
    if any(p["dev_pct_p50"] > DEVIATION_LIMIT_PCT for p in points):
        st.warning("Median forecast crosses the spec limit at one or more horizons.")

# ── Recommendations: retrieval (k-NN) vs optimization (SLSQP) ───────
st.divider()
st.subheader("Recommended corrective setpoints — retrieval vs optimization")
if risk > 0.35:
    col_retrieval, col_opt = st.columns(2)

    with col_retrieval:
        st.markdown("#### Retrieval (k-NN)")
        recs, neighbors = recommender.recommend(feats, new_corr_tags)
        st.caption(f"Nearest in-spec historical transitions: episodes {neighbors}")
        for r in recs:
            rec_id = f"{eid}-{t_now}-{r.tag}"
            col_a, col_b, col_c = st.columns([4, 1, 1])
            with col_a:
                st.markdown(f"**{r.tag} → {r.value:.0f}**  {r.primary_tag}")
                st.caption(r.reason)
            with col_b:
                if st.button("Accept", key=f"a-{rec_id}"):
                    log_feedback(new_recommendation_id(), eid, r.tag, r.value,
                                r.primary_tag, True)
                    st.success("Logged")
            with col_c:
                if st.button("Reject", key=f"r-{rec_id}"):
                    log_feedback(new_recommendation_id(), eid, r.tag, r.value,
                                r.primary_tag, False)
                    st.warning("Logged")

    with col_opt:
        st.markdown("#### Optimization (scipy SLSQP)")
        opt = optimize_setpoints(bundle, feats)
        st.caption(f"Predicted risk: {opt['risk_before']:.0%} → "
                  f"{opt['risk_after']:.0%} "
                  f"({opt['risk_before'] - opt['risk_after']:+.0%})")
        if not opt["setpoints"]:
            st.write("No actuator setpoints available to optimize.")
        else:
            for sp in opt["setpoints"]:
                bound_note = " *(at recipe limit)*" if sp["at_recipe_bound"] else ""
                st.markdown(f"**{sp['tag']}**: {sp['current']:.0f} → "
                           f"{sp['optimized']:.0f}{bound_note}")
            opt_rec_id = f"{eid}-{t_now}-optimizer"
            col_x, col_y = st.columns(2)
            with col_x:
                if st.button("Accept optimizer plan", key=f"a-{opt_rec_id}"):
                    for sp in opt["setpoints"]:
                        log_feedback(new_recommendation_id(), eid, sp["tag"],
                                    sp["optimized"], "[Optimizer]", True)
                    st.success("Logged")
            with col_y:
                if st.button("Reject optimizer plan", key=f"r-{opt_rec_id}"):
                    for sp in opt["setpoints"]:
                        log_feedback(new_recommendation_id(), eid, sp["tag"],
                                    sp["optimized"], "[Optimizer]", False)
                    st.warning("Logged")
else:
    st.info("Risk below action threshold — no corrective action recommended.")

# ── Correlation discovery panel ─────────────────────────────────────
st.divider()
st.subheader("Newly discovered correlations (not in configured loops)")
if new_corr.empty:
    st.write("None above threshold.")
else:
    st.dataframe(new_corr, use_container_width=True)
    st.caption("These relationships influence basis weight but are absent "
               "from the QCS-configured control loops.")

# ── Feedback / accuracy tracker ─────────────────────────────────────
st.divider()
st.subheader("Suggestion accuracy tracker")
acc = accuracy_summary()
if acc.empty:
    st.write("No operator feedback logged yet.")
else:
    st.dataframe(acc, use_container_width=True)
    with st.expander("Full feedback log"):
        st.dataframe(feedback_log(), use_container_width=True)

st.markdown("#### Feedback loop: accept-rate trend")
trend_df = accept_rate_trend()
if trend_df.empty:
    st.write("No feedback logged yet — accept/reject a few recommendations above to seed this chart.")
else:
    st.line_chart(trend_df.set_index("ts")["rolling_accept_rate"])
    st.caption("Rolling 20-recommendation acceptance rate. Rejected retrieval "
              "neighbors are down-weighted in src/recommender.py — a rising "
              "trend here is the signal that's working.")