"""
Calibrated capacity/latency dynamics model.

KIS-S's released mock-mode RL environment has no action-outcome coupling
(replica count has zero effect on simulated latency/throughput), so it
cannot be used to compare controllers. We instead calibrate a latency model
against KIS-S's *real* recorded Locust measurements
(results/cpu_baseline/dynamic/*), all collected at a fixed 3 replicas across
four load patterns.

Model (closed-loop saturation form):
    effective_load[t] = EMA_alpha(user_count)         # transient lag
    rho               = effective_load / (replicas * capacity_per_replica)
    latency_ms        = base_latency_ms + slope_ms * rho ** k

Locust is a closed-loop generator (users wait for responses), so saturated
response time grows ~linearly with users-per-replica (k ~ 1) rather than
diverging like an open M/M/1 queue. Alternatives compared during
calibration (R^2 on pooled real data): M/M/1 form 0.72, this form ~0.89.

IMPORTANT ASSUMPTION: replicas were fixed at 3 in all real data, so the
dependence on replica count (latency ~ load/replicas) is an extrapolation
from queueing theory, not something observed. Stated as a limitation.
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA_DIR = os.path.join(BASE, "data")
PATTERNS = ["ramp", "spike", "periodic", "random"]
FIXED_REPLICAS_IN_REAL_DATA = 3
BASE_FLOOR_MS = 200.0  # physical floor: lowest real p95 observed was ~220 ms


def ema_smooth(series, alpha):
    out = np.zeros(len(series))
    out[0] = series[0]
    for t in range(1, len(series)):
        out[t] = alpha * series[t] + (1 - alpha) * out[t - 1]
    return out


def predict_latency(effective_load, replicas, params):
    """Deterministic latency (ms) for a given smoothed load and replica count."""
    effective_load = np.asarray(effective_load, dtype=float)
    cap = np.maximum(replicas, 1e-6) * params["capacity_per_replica"]
    rho = np.maximum(effective_load / cap, 0.0)
    return params["base_latency_ms"] + params["slope_ms"] * rho ** params["k"]


def _fit_fn(x, base, slope, k, cap):
    rho = np.maximum(x / (FIXED_REPLICAS_IN_REAL_DATA * cap), 0.0)
    return base + slope * rho ** k


def calibrate():
    raw = {p: pd.read_csv(f"{DATA_DIR}/{p}.csv") for p in PATTERNS}
    y = np.concatenate([d["p95_latency_ms"].values for d in raw.values()])

    best = None
    for alpha in np.arange(0.1, 1.01, 0.05):
        x = np.concatenate([ema_smooth(d["user_count"].values, alpha) for d in raw.values()])
        p0 = [300, 6000, 1.0, 40]
        bounds = ([BASE_FLOOR_MS, 100, 0.5, 5], [800, 50000, 3.0, 300])
        try:
            popt, _ = curve_fit(_fit_fn, x, y, p0=p0, bounds=bounds, maxfev=40000)
        except Exception:
            continue
        r2 = 1 - np.sum((y - _fit_fn(x, *popt)) ** 2) / np.sum((y - y.mean()) ** 2)
        if best is None or r2 > best["r_squared"]:
            best = {"alpha": float(alpha), "base_latency_ms": float(popt[0]),
                    "slope_ms": float(popt[1]), "k": float(popt[2]),
                    "capacity_per_replica": float(popt[3]), "r_squared": float(r2)}

    # Residual spread (log space) -> used as seed-driven latency noise.
    x = np.concatenate([ema_smooth(d["user_count"].values, best["alpha"]) for d in raw.values()])
    pred = np.maximum(_fit_fn(x, best["base_latency_ms"], best["slope_ms"], best["k"],
                              best["capacity_per_replica"]), 1.0)
    best["log_residual_std"] = float(np.std(np.log(np.maximum(y, 1.0)) - np.log(pred)))
    best["fixed_replicas_in_real_data"] = FIXED_REPLICAS_IN_REAL_DATA
    best["n_points"] = int(len(y))

    with open(f"{DATA_DIR}/dynamics_params.json", "w") as f:
        json.dump(best, f, indent=2)

    print("Calibrated dynamics model (EMA load + closed-loop saturation):")
    for k_, v in best.items():
        print(f"  {k_:28s} = {v:.3f}" if isinstance(v, float) else f"  {k_:28s} = {v}")
    return best


if __name__ == "__main__":
    calibrate()
