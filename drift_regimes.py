"""
Drift-regime load traces, grounded in the *shapes* of KIS-S's real load
patterns (offered load = concurrent users):

  stationary : control, no drift (synthetic by definition)
  sudden     : real 'spike' shape  -> abrupt low->high step
  gradual    : real 'ramp' shape   -> smooth interpolated rise
  recurring  : periodic alternation low/high (square wave, per proposal)
  cyclic     : real 'periodic' shape -> smooth sinusoid (supplementary)

NOTE on noise: in the real data, user_count is a *commanded* Locust
parameter (deterministic), so there is no real load jitter to resample.
We add small Gaussian jitter (JITTER_SD users, seeded) so that ADWIN does
not operate on perfectly constant signals and seeds affect the load too.
This jitter is synthetic and stated as such.
"""
import numpy as np
import pandas as pd

from config import DATA_DIR, FIGURES_DIR

EPISODE_LEN = 60
LOW_LOAD = 15.0
HIGH_LOAD = 90.0
MID_LOAD = 52.0
JITTER_SD = 2.0

# Step at which drift begins (None = no single onset).
RECUR_HALF_PERIOD = 15
DRIFT_ONSET = {"stationary": None, "sudden": 30, "gradual": 18,
               "recurring": RECUR_HALF_PERIOD, "cyclic": None}
# First four = the proposal's regimes. 'cyclic' (smooth sinusoid taken
# directly from the real 'periodic' pattern) is a supplementary regime.
REGIMES = ["stationary", "sudden", "gradual", "recurring", "cyclic"]
MAIN_REGIMES = REGIMES[:4]


def _shape(pattern):
    return pd.read_csv(f"{DATA_DIR}/{pattern}.csv")["user_count"].values.astype(float)


def _resample(values, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(values)), values)


def build_regime(name, seed):
    rng = np.random.default_rng(seed)
    n = EPISODE_LEN
    if name == "stationary":
        base = np.full(n, MID_LOAD)
    elif name == "sudden":
        base = np.where(np.arange(n) < DRIFT_ONSET["sudden"], LOW_LOAD, HIGH_LOAD)
    elif name == "gradual":
        ramp = _shape("ramp")
        ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
        start, end = DRIFT_ONSET["gradual"], int(n * 0.7)
        base = np.empty(n)
        base[:start] = LOW_LOAD
        base[start:end] = LOW_LOAD + (HIGH_LOAD - LOW_LOAD) * _resample(ramp, end - start)
        base[end:] = HIGH_LOAD
    elif name == "recurring":
        # periodic alternation between two regimes (square wave), as defined
        # in the proposal: low, high, low, high, each RECUR_HALF_PERIOD steps
        phase = (np.arange(n) // RECUR_HALF_PERIOD) % 2
        base = np.where(phase == 0, LOW_LOAD, HIGH_LOAD)
    elif name == "cyclic":
        base = _resample(_shape("periodic"), n)
    else:
        raise ValueError(name)
    return np.clip(base + rng.normal(0, JITTER_SD, n), 3.0, 120.0)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax, name in zip(axes.flat, REGIMES):
        for s in range(3):
            ax.plot(build_regime(name, s), alpha=0.7, label=f"seed {s}")
        if DRIFT_ONSET[name] is not None:
            ax.axvline(DRIFT_ONSET[name], color="k", ls=":", lw=1)
        ax.set_title(name); ax.set_xlabel("step"); ax.set_ylabel("offered load (users)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/drift_regimes.png", dpi=130)
    print("saved")
