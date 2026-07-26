"""Grade Change Intelligence — operator dashboard."""
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (ACTUATOR_TAGS, DATA_PATH, DEVIATION_LIMIT_PCT,
                        META_PATH, MODEL_DIR, QUALITY_TAG, TRIGGER_STEP)
from src.correlations import discover_new_correlations
from src.features import episode_features
from src.feedback import (accuracy_summary, feedback_log, log_feedback,
                          new_recommendation_id)
from src.model import load_risk_model, predict_risk, shap_top_drivers
from src.recommender import SetpointRecommender

st.set_page_config(page_title="Grade Change Intelligence", layout="wide")


@st.cache_resource
def load_all():
    episodes = pd.read_parquet(DATA_PATH)
    meta = pd.read_csv(META_PATH)
    bundle = load_risk_model(MODEL_DIR)
    table = pd.read_parquet(f"{MODEL_DIR}/feature_table.parquet")
    rec = SetpointRecommender(table, meta, episodes)
    return episodes, meta, bundle, rec


episodes, meta, bundle, recommender = load_all()

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

# ── Recommendations ─────────────────────────────────────────────────
st.divider()
st.subheader("Recommended corrective setpoints")
if risk > 0.35:
    recs, neighbors = recommender.recommend(feats, new_corr_tags)
    st.caption(f"Based on nearest in-spec historical transitions: "
               f"episodes {neighbors}")
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
