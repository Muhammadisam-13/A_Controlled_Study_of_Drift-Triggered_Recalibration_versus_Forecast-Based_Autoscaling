"""
Capacity-drift extension.

The load-level experiments (simulator.py) only change *how much traffic*
arrives. A controller that sizes the fleet from a capacity model is
unaffected by that kind of drift as long as the model stays right.
Capacity drift breaks the model itself: the useful capacity of one replica
changes (noisy neighbours, CPU throttling, a slower model version, ...).

    true capacity per replica at step t = capacity_per_replica * m(t)
    m(t) = 1.0 before drift; < 1 when replicas become less capable

(IMPORTANT: no real KIS-S measurement contains a capacity change -- all runs
used fixed hardware. The capacity multiplier m(t) is a synthetic scenario,
applied to the capacity parameter of the model calibrated on real data.)

Controllers (all see the same load, latency, effective replicas):
  S   step rule on observed latency, model-free after its initial anchor
  RF  HPA-style proportional controller, capacity belief FIXED
  RR  same, capacity belief RECALIBRATED after ADWIN fires on the residual
  RO  same, ORACLE: knows true capacity each step (upper bound)
  PF  Holt forecast + capacity planning, belief FIXED   (= arm B of the 2x2)
  PR  same, RECALIBRATED
  PO  same, ORACLE

Recalibration (RR, PR):
  residual_t = log(observed latency) - log(model-predicted latency | believed capacity)
  ADWIN watches the residual stream. When it fires, ADWIN's remaining
  window gives the change point; the capacity belief is re-fit by 1-D least
  squares (log-latency) on observations since the change point, and keeps
  being updated each step until REFIT_WINDOW observations have accumulated.
"""
import numpy as np
import pandas as pd

from dynamics import predict_latency
from drift_regimes import EPISODE_LEN, JITTER_SD
from controllers import (ReactiveController, ProactiveController, make_adwin,
                         STARTUP_DELAY_STEPS, SLO_THRESHOLD_MS, REFIT_WINDOW)
from simulator import load_dynamics_params, CALIB_STEPS

CAP_LOAD = 40.0          # constant offered load (users) in capacity-drift regimes
CAP_DROP = 0.55          # capacity multiplier after drift
CAP_REGIMES = ["cap_stationary", "cap_sudden", "cap_gradual", "cap_recurring"]
CAP_ONSET = {"cap_stationary": None, "cap_sudden": 20, "cap_gradual": 15, "cap_recurring": 15}
CAP_ARMS = ["S", "RF", "RR", "RO", "PF", "PR", "PO"]
CAP_GRID = np.arange(4.0, 120.01, 0.5)
MIN_FIT_OBS = 4
RESIDUAL_ADWIN_DELTA = 0.002   # the delta used for the load-drift experiments
CUSUM_K, CUSUM_H = 0.5, 6.0    # allowance / threshold in residual-std units; H tuned on no-drift noise only (4% false alarm per 60-step episode)
DETECTOR = {"name": "cusum", "adwin_delta": RESIDUAL_ADWIN_DELTA}   # module-level switch


def capacity_multiplier(regime):
    n = EPISODE_LEN
    t = np.arange(n)
    m = np.ones(n)
    if regime == "cap_sudden":
        m[t >= 20] = CAP_DROP
    elif regime == "cap_gradual":
        frac = np.clip((t - 15) / (40 - 15), 0, 1)
        m = 1 - (1 - CAP_DROP) * frac
    elif regime == "cap_recurring":
        m = np.where((t // 15) % 2 == 0, 1.0, CAP_DROP)
    return m


class CapacityBelief:
    """Capacity estimate + ADWIN-on-residual recalibration (shared by RR, PR)."""

    def __init__(self, dyn, recalibrate):
        self.dyn = dict(dyn)
        self.cap_hat = float(dyn["capacity_per_replica"])
        self.recalibrate = recalibrate
        self.detector = DETECTOR["name"]
        self.adwin = make_adwin(DETECTOR["adwin_delta"]) if recalibrate and self.detector == "adwin" else None
        self.sigma = float(dyn["log_residual_std"])
        self.s_hi = self.s_lo = 0.0
        self.ema = None
        self.obs = []                 # (ema_load, effective_replicas, latency)
        self.segment_start = None     # index into obs where post-change data begins
        self.detection_steps = []
        self.n_adaptations = 0

    def observe(self, t, load, latency, eff):
        a = self.dyn["alpha"]
        self.ema = load if self.ema is None else a * load + (1 - a) * self.ema
        self.obs.append((self.ema, eff, latency))
        if not self.recalibrate:
            return False
        d = dict(self.dyn); d["capacity_per_replica"] = self.cap_hat
        resid = np.log(latency) - np.log(float(predict_latency(self.ema, eff, d)))
        if self.detector == "adwin":
            self.adwin.update(resid)
            fired = bool(self.adwin.drift_detected)
            width = int(max(self.adwin.width, MIN_FIT_OBS)) if fired else 0
        else:   # two-sided CUSUM on the standardised residual
            z = resid / self.sigma
            self.s_hi = max(0.0, self.s_hi + z - CUSUM_K)
            self.s_lo = max(0.0, self.s_lo - z - CUSUM_K)
            fired = self.s_hi > CUSUM_H or self.s_lo > CUSUM_H
            width = 0
            if fired:
                # change point ~ when the winning statistic last left zero
                run = 0
                for (e, r_, l_) in reversed(self.obs):
                    run += 1
                    dd = dict(self.dyn); dd["capacity_per_replica"] = self.cap_hat
                    zz = (np.log(l_) - np.log(float(predict_latency(e, r_, dd)))) / self.sigma
                    if (self.s_hi > CUSUM_H and zz < 0) or (self.s_lo > CUSUM_H and zz > 0):
                        break
                width = int(min(max(run, MIN_FIT_OBS), len(self.obs)))
                self.s_hi = self.s_lo = 0.0
        if fired:
            self.detection_steps.append(t)
            self.segment_start = max(len(self.obs) - width, 0)
        if self.segment_start is not None:
            lo = max(self.segment_start, len(self.obs) - REFIT_WINDOW)
            if len(self.obs) - lo >= MIN_FIT_OBS:
                self._refit(self.obs[lo:])
                self.n_adaptations += 1
            if len(self.obs) - self.segment_start >= REFIT_WINDOW:
                self.segment_start = None
        return fired

    def _refit(self, window):
        ema = np.array([w[0] for w in window]); rep = np.array([w[1] for w in window], float)
        y = np.log(np.array([w[2] for w in window]))
        best, best_sse = self.cap_hat, np.inf
        for c in CAP_GRID:
            d = dict(self.dyn); d["capacity_per_replica"] = c
            pred = np.log(np.maximum(predict_latency(ema, rep, d), 1.0))
            sse = float(np.sum((y - pred) ** 2))
            if sse < best_sse:
                best, best_sse = c, sse
        self.cap_hat = float(best)


class _CapMixin:
    """Wires a CapacityBelief into an existing controller (replaces its .dyn)."""
    mode_name = None

    def _init_belief(self, dyn, recalibrate, oracle_caps=None):
        self.belief = CapacityBelief(dyn, recalibrate)
        self.oracle_caps = oracle_caps
        self.detection_steps = self.belief.detection_steps
        self.n_adaptations = 0

    def _sync(self, t):
        if self.oracle_caps is not None:
            self.belief.cap_hat = float(self.oracle_caps[min(t + 1, len(self.oracle_caps) - 1)])
        d = dict(self.belief.dyn); d["capacity_per_replica"] = self.belief.cap_hat
        self.dyn = d


class CapReactive(_CapMixin, ReactiveController):
    def __init__(self, dyn, calib, recalibrate=False, oracle_caps=None):
        ReactiveController.__init__(self, dyn, adaptive=False, calib_window=calib, mode="proportional")
        self._init_belief(dyn, recalibrate, oracle_caps)

    def decide(self, t, load, latency, recent_loads, eff):
        self.belief.observe(t, load, latency, eff)
        self._sync(t)
        self.n_adaptations = self.belief.n_adaptations
        return ReactiveController.decide(self, t, load, latency, recent_loads, eff)


class CapProactive(_CapMixin, ProactiveController):
    def __init__(self, dyn, calib, recalibrate=False, oracle_caps=None):
        ProactiveController.__init__(self, dyn, adaptive=False, calib_window=calib)
        self._init_belief(dyn, recalibrate, oracle_caps)

    def decide(self, t, load, latency, recent_loads, eff):
        self.belief.observe(t, load, latency, eff)
        self._sync(t)
        self.n_adaptations = self.belief.n_adaptations
        return ProactiveController.decide(self, t, load, latency, recent_loads, eff)


def make_cap_controller(arm, dyn, calib, true_caps):
    if arm == "S":
        c = ReactiveController(dyn, adaptive=False, calib_window=calib, mode="step")
        c.belief = None
        return c
    if arm == "RF": return CapReactive(dyn, calib)
    if arm == "RR": return CapReactive(dyn, calib, recalibrate=True)
    if arm == "RO": return CapReactive(dyn, calib, oracle_caps=true_caps)
    if arm == "PF": return CapProactive(dyn, calib)
    if arm == "PR": return CapProactive(dyn, calib, recalibrate=True)
    if arm == "PO": return CapProactive(dyn, calib, oracle_caps=true_caps)
    raise ValueError(arm)


def build_cap_load(seed):
    rng = np.random.default_rng(50_000 + seed)
    return np.clip(CAP_LOAD + rng.normal(0, JITTER_SD, EPISODE_LEN), 3.0, 120.0)


def _longest_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def run_cap_episode(arm, regime, seed, dyn=None, keep_records=False):
    dyn = dyn or load_dynamics_params()
    load = build_cap_load(seed)
    mult = capacity_multiplier(regime)
    true_caps = dyn["capacity_per_replica"] * mult
    noise_rng = np.random.default_rng(20_000 + seed * 10 + CAP_REGIMES.index(regime))
    eps = noise_rng.normal(0.0, dyn["log_residual_std"], EPISODE_LEN)   # shared across arms

    ctl = make_cap_controller(arm, dyn, load[:CALIB_STEPS], true_caps)
    initial = ctl.replicas
    decided, lat, reps, viol, caphat = [], [], [], [], []
    ema = load[0]
    for t in range(EPISODE_LEN):
        ema = dyn["alpha"] * load[t] + (1 - dyn["alpha"]) * ema
        eff = initial if t - STARTUP_DELAY_STEPS < 0 else decided[t - STARTUP_DELAY_STEPS]
        d = dict(dyn); d["capacity_per_replica"] = true_caps[t]
        latency = float(predict_latency(ema, eff, d)) * float(np.exp(eps[t]))
        lat.append(latency); reps.append(eff); viol.append(latency > SLO_THRESHOLD_MS)
        decided.append(ctl.decide(t, load[t], latency, list(load[:t + 1]), eff))
        caphat.append(ctl.belief.cap_hat if getattr(ctl, "belief", None) else np.nan)

    viol = np.array(viol); onset = CAP_ONSET[regime]
    post = viol[onset:] if onset is not None else None
    det = list(ctl.detection_steps)
    det_after = [x for x in det if onset is not None and x >= onset]
    caphat = np.array(caphat)
    cap_err = np.nan
    if onset is not None and not np.all(np.isnan(caphat)):
        cap_err = float(np.nanmean(np.abs(caphat[onset:] - true_caps[onset:]) / true_caps[onset:]))
    out = {
        "arm": arm, "regime": regime, "seed": seed,
        "cost": int(np.sum(reps)),
        "slo_violation_rate": float(viol.mean()),
        "post_drift_violation_rate": float(post.mean()) if post is not None else np.nan,
        "longest_violation_run": int(_longest_run(post if post is not None else viol)),
        "churn": int(np.sum(np.abs(np.diff(reps)))),
        "n_detections": len(det),
        "detection_lag": (det_after[0] - onset) if det_after else np.nan,
        "false_detections": len([x for x in det if onset is not None and x < onset]) if onset is not None else len(det),
        "capacity_rel_error_post": cap_err,
    }
    if keep_records:
        out["records"] = {"replicas": [int(r) for r in reps], "latency": lat,
                          "cap_hat": caphat.tolist(), "true_cap": true_caps.tolist(),
                          "violation": viol.tolist()}
    return out


if __name__ == "__main__":
    for regime in CAP_REGIMES:
        for arm in CAP_ARMS:
            r = run_cap_episode(arm, regime, seed=0)
            print(f"{regime:15s} {arm:2s} cost={r['cost']:4d} viol={r['slo_violation_rate']:.2f} "
                  f"post={r['post_drift_violation_rate']:.2f} det={r['n_detections']} "
                  f"lag={r['detection_lag']} capErr={r['capacity_rel_error_post']:.2f}")
