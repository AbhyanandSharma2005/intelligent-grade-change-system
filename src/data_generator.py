"""Synthetic QCS/DCS historian: grade-change episodes as multivariate
time series, with a planted hidden correlation (dryer humidity couples
into moisture and, through it, basis weight) that is NOT in any
configured control loop — for the correlation-discovery engine to find."""
import numpy as np
import pandas as pd

from src.config import (ACTUATOR_TAGS, DEVIATION_LIMIT_PCT, GRADES,
                        QUALITY_TAG, STABLE_WINDOW_STEPS, STEPS_PER_EPISODE,
                        TRIGGER_STEP)


def _ramp(start, end, n_total, trigger, ramp_len, rng):
    """Setpoint profile: flat, then linear ramp with actuator lag + noise."""
    sp = np.full(n_total, float(start))
    end_idx = min(trigger + ramp_len, n_total)
    sp[trigger:end_idx] = np.linspace(start, end, end_idx - trigger)
    sp[end_idx:] = end
    actual = np.zeros(n_total)
    actual[0] = start
    for t in range(1, n_total):
        actual[t] = actual[t - 1] + 0.25 * (sp[t] - actual[t - 1])
    return actual + rng.normal(0, 0.002 * abs(end), n_total)


def simulate_episode(episode_id: int, rng: np.random.Generator,
                     force_failure: bool = False):
    src, dst = rng.choice(list(GRADES.keys()), size=2, replace=False)
    g0, g1 = GRADES[src], GRADES[dst]
    n, trig = STEPS_PER_EPISODE, TRIGGER_STEP
    ramp_len = int(rng.uniform(60, 140))

    tags = {a: _ramp(g0[a], g1[a], n, trig, ramp_len, rng)
            for a in ACTUATOR_TAGS}

    # Hidden exogenous variable: dryer humidity (slow random walk).
    # In "failure" episodes it drifts strongly mid-transition.
    hum = 45 + np.cumsum(rng.normal(0, 0.05, n))
    if force_failure:
        drift = np.zeros(n)
        drift[trig:trig + 150] = np.linspace(0, rng.uniform(8, 15), 150)
        drift[trig + 150:] = drift[trig + 149]
        hum = hum + drift
    tags["dryer_humidity"] = hum

    # Moisture: driven by steam pressure (configured loop) AND humidity (hidden)
    moisture = (7.5 - 0.009 * tags["steam_pressure"]
                + 0.045 * (hum - 45) + rng.normal(0, 0.05, n))
    tags["moisture"] = moisture

    # Ash: filler flow loop
    tags["ash"] = 0.028 * tags["filler_flow"] + rng.normal(0, 0.15, n)

    # Basis weight: configured physics (stock flow, speed) + hidden
    # moisture coupling + first-order lag + noise
    bw_ss = (0.0122 * tags["stock_flow"] - 0.021 * tags["machine_speed"]
             + 0.003 * tags["filler_flow"] + 0.9 * (moisture - 4.5) + 32.0)
    bw = np.zeros(n)
    bw[0] = g0["bw_setpoint"]
    for t in range(1, n):
        bw[t] = bw[t - 1] + (bw_ss[t] - bw[t - 1]) / 10.0
    # rescale so nominal transitions land near setpoints
    bw = bw - bw[trig - 1] + g0["bw_setpoint"]
    scale = (g1["bw_setpoint"] - g0["bw_setpoint"]) / max(bw[-1] - bw[0], 1e-6)
    bw = g0["bw_setpoint"] + (bw - bw[0]) * scale
    bw += 0.9 * (moisture - moisture[trig]) * (np.arange(n) >= trig)
    bw += rng.normal(0, 0.12, n)
    tags[QUALITY_TAG] = bw

    # Caliper: correlated with bw
    tags["caliper"] = bw * 1.28 + rng.normal(0, 0.4, n)

    df = pd.DataFrame(tags)
    df["step"] = np.arange(n)
    df["episode_id"] = episode_id

    # Labels
    sp_traj = np.full(n, g0["bw_setpoint"])
    sp_traj[trig:] = np.interp(np.arange(trig, n), [trig, trig + ramp_len],
                               [g0["bw_setpoint"], g1["bw_setpoint"]])
    sp_traj = np.clip(sp_traj, min(g0["bw_setpoint"], g1["bw_setpoint"]),
                      max(g0["bw_setpoint"], g1["bw_setpoint"]))
    df["bw_setpoint"] = sp_traj
    dev_pct = np.abs(bw - sp_traj) / sp_traj * 100
    post = dev_pct[trig + ramp_len:]
    off_spec = bool((post > DEVIATION_LIMIT_PCT).any())

    in_band = dev_pct <= DEVIATION_LIMIT_PCT
    stab_step = n
    for t in range(trig + ramp_len, n - STABLE_WINDOW_STEPS):
        if in_band[t:t + STABLE_WINDOW_STEPS].all():
            stab_step = t
            break

    meta = {"episode_id": episode_id, "from_grade": src, "to_grade": dst,
            "ramp_len": ramp_len, "off_spec": off_spec,
            "stabilization_step": stab_step,
            "time_to_stabilize_s": (stab_step - trig) * 5}
    return df, meta


def generate_dataset(n_episodes: int = 400, failure_rate: float = 0.25,
                     seed: int = 42):
    rng = np.random.default_rng(seed)
    frames, metas = [], []
    for i in range(n_episodes):
        df, meta = simulate_episode(i, rng,
                                    force_failure=rng.random() < failure_rate)
        frames.append(df)
        metas.append(meta)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metas)
