"""
Discrete-time autoscaling simulator.

At step t:
  1. Offered load load[t] arrives (from the drift-regime trace).
  2. Effective replicas = the decision made STARTUP_DELAY_STEPS ago
     (pods take time to start; capacity changes are not instantaneous).
  3. Latency = calibrated model(EMA(load), effective replicas) * lognormal
     noise. The noise sequence is shared across arms for a given
     (regime, seed) -> paired comparisons (common random numbers).
  4. Cost accrues as replica-steps; SLO violation if latency > threshold.
  5. The controller observes (load, latency) and issues a new decision.
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import json
import numpy as np

from dynamics import predict_latency
from drift_regimes import build_regime, EPISODE_LEN, DRIFT_ONSET, REGIMES
from controllers import (ReactiveController, ProactiveController,
                         STARTUP_DELAY_STEPS, SLO_THRESHOLD_MS)

DATA_DIR = os.path.join(BASE, "data")
CALIB_STEPS = 10   # first N steps of each trace used to calibrate/fit every arm


def load_dynamics_params():
    with open(f"{DATA_DIR}/dynamics_params.json") as f:
        return json.load(f)


def make_controller(arm, dyn, calib, reactive_mode="step"):
    if arm == "A":
        return ReactiveController(dyn, adaptive=False, calib_window=calib, mode=reactive_mode)
    if arm == "D":
        return ReactiveController(dyn, adaptive=True, calib_window=calib, mode=reactive_mode)
    if arm == "B":
        return ProactiveController(dyn, adaptive=False, calib_window=calib)
    if arm == "C":
        return ProactiveController(dyn, adaptive=True, calib_window=calib)
    raise ValueError(arm)


def _longest_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def run_episode(arm, regime, seed, dyn=None, reactive_mode="step", keep_records=False):
    dyn = dyn or load_dynamics_params()
    load = build_regime(regime, seed)
    noise_rng = np.random.default_rng(10_000 + seed * 10 + REGIMES.index(regime))
    eps = noise_rng.normal(0.0, dyn["log_residual_std"], EPISODE_LEN)

    ctl = make_controller(arm, dyn, load[:CALIB_STEPS], reactive_mode)
    initial_replicas = ctl.replicas
    decided = []                      # decided[t] = decision issued at end of step t
    ema = load[0]
    lat, reps, viol = [], [], []

    for t in range(EPISODE_LEN):
        ema = dyn["alpha"] * load[t] + (1 - dyn["alpha"]) * ema
        eff = initial_replicas if t - STARTUP_DELAY_STEPS < 0 else decided[t - STARTUP_DELAY_STEPS]
        latency = float(predict_latency(ema, eff, dyn)) * float(np.exp(eps[t]))
        lat.append(latency); reps.append(eff); viol.append(latency > SLO_THRESHOLD_MS)
        decided.append(ctl.decide(t, load[t], latency, list(load[:t + 1]), eff))

    viol = np.array(viol)
    onset = DRIFT_ONSET[regime]
    post = viol[onset:] if onset is not None else None
    det_after = [d for d in ctl.detection_steps if onset is not None and d >= onset]
    out = {
        "arm": arm, "regime": regime, "seed": seed, "reactive_mode": reactive_mode,
        "cost": int(np.sum(reps)),
        "slo_violation_rate": float(viol.mean()),
        "post_drift_violation_rate": float(post.mean()) if post is not None else np.nan,
        "longest_violation_run": int(_longest_run(post if post is not None else viol)),
        "churn": int(np.sum(np.abs(np.diff(reps)))),
        "n_adaptations": int(ctl.n_adaptations),
        "n_detections": len(ctl.detection_steps),
        "detection_lag": (det_after[0] - onset) if det_after else np.nan,
        "false_detections": len([d for d in ctl.detection_steps if onset is not None and d < onset]),
    }
    if keep_records:
        out["records"] = {"load": load.tolist(), "replicas": [int(r) for r in reps],
                          "latency": lat, "violation": viol.tolist()}
    return out


if __name__ == "__main__":
    for regime in REGIMES:
        for arm in "ABCD":
            r = run_episode(arm, regime, seed=0)
            print(f"{regime:10s} {arm} cost={r['cost']:4d} viol={r['slo_violation_rate']:.2f} "
                  f"post={r['post_drift_violation_rate']:.2f} run={r['longest_violation_run']:2d} "
                  f"churn={r['churn']:2d} adapt={r['n_adaptations']} lag={r['detection_lag']} "
                  f"falseDet={r['false_detections']}")
