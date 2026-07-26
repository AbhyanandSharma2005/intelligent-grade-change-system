# Grade Change Intelligence — Paper Making MVP

A predictive + explainable co-pilot for MD grade transitions. Sits on top of
an existing QCS: predicts basis-weight deviation risk (>±2.5% of setpoint)
during a live grade change, recommends corrective setpoints with a tagged
rationale, discovers correlations not in the configured control loops, and
logs operator accept/reject feedback.

## Run (3 commands)
```bash
pip install -r requirements.txt
python scripts/generate_data.py     # synthetic historian: 400 episodes
python scripts/train.py             # XGBoost risk model + artifacts
streamlit run app.py                # dashboard
