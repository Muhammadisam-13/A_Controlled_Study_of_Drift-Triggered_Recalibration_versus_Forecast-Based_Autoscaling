"""
The four controllers in the 2x2 factorial design:

                Static                        Drift-triggered adaptation
Reactive   | A: threshold rule, anchor     | D: same rule; on ADWIN detection
(threshold)|    calibrated once             |    re-derives anchor + resets
Proactive  | B: Holt forecaster, params    | C: same forecaster; on ADWIN
(forecast) |    estimated once, state      |    detection RE-ESTIMATES params on
           |    keeps filtering new data   |    the last REFIT_WINDOW points

Fairness notes
  * Static arms are NOT frozen strawmen: B keeps updating its level/trend
    state with every new observation (standard Holt filtering) using
    parameters fitted once. Only parameter re-estimation is withheld.
  * Arms C and D use the identical ADWIN configuration and the same
    REFIT_WINDOW of recent observations on detection.
  * All arms share the same capacity-planning model (the calibrated
    dynamics model) and the same provisioning headroom (TARGET_FRACTION).
    I.e. controllers are assumed to know the system's capacity curve, as
    after operator load-testing; what differs is how they track demand.
"""
import warnings
import numpy as np

from dynamics import predict_latency

warnings.filterwarnings("ignore")

MIN_REPLICAS = 1
MAX_REPLICAS = 10
STARTUP_DELAY_STEPS = 2
SLO_THRESHOLD_MS = 3000.0
TARGET_FRACTION = 0.5      # provision so predicted latency <= 50% of SLO (headroom for noise)
UP_FRAC, DOWN_FRAC = 0.8, 0.25   # reactive thresholds as fractions of SLO
COOLDOWN_STEPS = 2
REFIT_WINDOW = 15
SAFETY_MARGIN = 1.15
HPA_TOLERANCE = 0.10


def make_adwin(delta=0.002):
    """Shared detector config for every adaptive arm.

    river's ADWIN defaults (clock=32, grace_period=10) only check for drift
    every 32 updates -- in a 60-step episode that is a single check, so it
    never fires. We check every step. Verified: detects a 15->90 step in
    ~2-3 steps with 0 false positives over 20 stationary runs.
    """
    try:
        from river import drift
    except ImportError as e:   # e.g. compiled extension blocked by OS policy
        raise ImportError(
            "ADWIN needs river (pip install river==0.26.1). If river's compiled "
            "extension is blocked on this machine (Windows Application Control), run "
            "in WSL/Colab, or use only the CUSUM capacity study "
            "(python run_capacity_experiment.py cusum-only).") from e
    return drift.ADWIN(delta=delta, clock=1, grace_period=5, min_window_length=3)


def anchor_replicas(loads, dyn):
    """Smallest replica count whose predicted latency at the 90th-percentile
    recent load is within the provisioning target."""
    target_load = float(np.percentile(loads, 90))
    for r in range(MIN_REPLICAS, MAX_REPLICAS + 1):
        if predict_latency(target_load, r, dyn) <= TARGET_FRACTION * SLO_THRESHOLD_MS:
            return r
    return MAX_REPLICAS


class ReactiveController:
    """mode='step': +/-1 replica threshold rule on smoothed observed latency.
       mode='proportional': HPA-style desired = ceil(current * rho/rho_target)
       (used as a sensitivity check on baseline strength)."""

    def __init__(self, dyn, adaptive, calib_window, mode="step"):
        self.dyn, self.adaptive, self.mode = dyn, adaptive, mode
        self.adwin = make_adwin() if adaptive else None
        self.replicas = anchor_replicas(calib_window, dyn)
        self.cooldown = 0
        self.lat_hist, self.load_hist = [], []
        self.n_adaptations = 0
        self.detection_steps = []
        # target utilisation (rho*) at which predicted latency == provisioning target
        tgt = TARGET_FRACTION * SLO_THRESHOLD_MS
        self.rho_star = ((tgt - dyn["base_latency_ms"]) / dyn["slope_ms"]) ** (1.0 / dyn["k"])

    def decide(self, t, load, latency, recent_loads, effective_replicas):
        self.lat_hist.append(latency)
        self.load_hist.append(load)
        if self.adaptive:
            self.adwin.update(load)
            if self.adwin.drift_detected:
                self.detection_steps.append(t)
                self.replicas = anchor_replicas(recent_loads[-REFIT_WINDOW:], self.dyn)
                self.cooldown = COOLDOWN_STEPS
                self.n_adaptations += 1
                return self.replicas

        if self.cooldown > 0:
            self.cooldown -= 1
            return self.replicas

        if self.mode == "proportional":
            ma_load = float(np.mean(self.load_hist[-3:]))
            rho = ma_load / (max(effective_replicas, 1) * self.dyn["capacity_per_replica"])
            ratio = rho / self.rho_star
            if abs(ratio - 1.0) > HPA_TOLERANCE:
                desired = int(np.ceil(effective_replicas * ratio))
                self.replicas = int(np.clip(desired, MIN_REPLICAS, MAX_REPLICAS))
                self.cooldown = COOLDOWN_STEPS
            return self.replicas

        ma_lat = float(np.mean(self.lat_hist[-3:]))
        if ma_lat > UP_FRAC * SLO_THRESHOLD_MS and self.replicas < MAX_REPLICAS:
            self.replicas += 1
            self.cooldown = COOLDOWN_STEPS
        elif ma_lat < DOWN_FRAC * SLO_THRESHOLD_MS and self.replicas > MIN_REPLICAS:
            self.replicas -= 1
            self.cooldown = COOLDOWN_STEPS
        return self.replicas


class ProactiveController:
    """Holt (additive-trend) forecaster, forecasting STARTUP_DELAY_STEPS ahead."""

    ALPHA_GRID = np.arange(0.1, 0.91, 0.1)
    BETA_GRID = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5])

    def __init__(self, dyn, adaptive, calib_window):
        self.dyn, self.adaptive = dyn, adaptive
        self.adwin = make_adwin() if adaptive else None
        self.n_adaptations = 0
        self.detection_steps = []
        self.params = None            # (alpha, beta)
        self.level = self.trend = None
        self._estimate_and_prime(np.asarray(calib_window, float))
        self.replicas = self._size()

    def _estimate_and_prime(self, series):
        """Bounded grid search of Holt smoothing parameters minimising
        one-step-ahead squared error on `series`, then run the recursion
        through `series` to set (level, trend).

        Bounds matter: unconstrained maximum-likelihood fitting on a flat
        calibration window returns alpha=0 (ignore all future data), which
        would make the static arm a strawman (observed in development).
        The same routine is used for the static arm's one-off fit and the
        adaptive arm's refits.
        """
        series = np.maximum(np.asarray(series, float), 1e-3)
        best = None
        for a in self.ALPHA_GRID:
            for b in self.BETA_GRID:
                lvl, tr, sse = series[0], 0.0, 0.0
                for x in series[1:]:
                    sse += (x - (lvl + tr)) ** 2
                    new_lvl = a * x + (1 - a) * (lvl + tr)
                    tr = b * (new_lvl - lvl) + (1 - b) * tr
                    lvl = new_lvl
                if best is None or sse < best[0]:
                    best = (sse, float(a), float(b), lvl, tr)
        _, a, b, lvl, tr = best
        self.params = (a, b)
        self.level, self.trend = lvl, tr

    def _filter(self, x):
        a, b = self.params
        prev = self.level
        self.level = a * x + (1 - a) * (self.level + self.trend)
        self.trend = b * (self.level - prev) + (1 - b) * self.trend

    def _forecast(self):
        return max(self.level + STARTUP_DELAY_STEPS * self.trend, 1.0)

    def _size(self):
        need = self._forecast() * SAFETY_MARGIN
        for r in range(MIN_REPLICAS, MAX_REPLICAS + 1):
            if predict_latency(need, r, self.dyn) <= TARGET_FRACTION * SLO_THRESHOLD_MS:
                return r
        return MAX_REPLICAS

    def decide(self, t, load, latency, recent_loads, effective_replicas):
        if self.adaptive:
            self.adwin.update(load)
            if self.adwin.drift_detected:
                self.detection_steps.append(t)
                self._estimate_and_prime(np.asarray(recent_loads[-REFIT_WINDOW:], float))
                self.n_adaptations += 1
                self.replicas = self._size()
                return self.replicas
        self._filter(load)
        self.replicas = self._size()
        return self.replicas
