# Adaptation or Anticipation? — Drift-triggered vs forecast-based autoscaling

2x2 factorial study (reactive/proactive x static/drift-triggered) built on real
load measurements released with **KIS-S** (Zhang et al., arXiv:2507.07932,
github.com/GuilinDev/KISim).

## Pipeline (run in this order)

```
pip install numpy pandas scipy matplotlib river
git clone https://github.com/GuilinDev/KISim ../KISim   # source of real traces
python extract_data.py      # KISim results/cpu_baseline/dynamic -> data/*.csv
python dynamics.py          # calibrate latency model -> data/dynamics_params.json
python drift_regimes.py     # build + plot the 5 regimes
python run_experiment.py    # 4 arms x 5 regimes x 30 seeds (+ HPA-style sensitivity)
python analyze.py           # tables, paired contrasts, figures
# --- capacity-drift extension (proposed contribution) ---
python capacity_drift.py            # smoke test, one seed
python run_capacity_experiment.py   # 7 arms x 4 regimes x 30 seeds (~10 s)
python analyze_capacity.py          # tables, paired contrasts, figures
```
All paths are centralized in `config.py` and resolved relative to the repo
itself, so no per-file edits are needed. The one machine-specific setting is
where KISim was cloned: by default `config.py` looks for it as a sibling
`../KISim` directory (matching the clone command above); override by setting
the `KISIM_ROOT` environment variable instead if you cloned it elsewhere.

## What was reproduced from KIS-S, and what was not

* Used: KIS-S's **real recorded Locust runs** (ramp / spike / periodic / random;
  fixed 3 replicas; per-step concurrent users + P95 latency) as ground truth,
  and its four load-pattern shapes (mapped to gradual / sudden / cyclic drift).
* **Not usable as released:** KISim's offline mock RL environment
  (`scripts/rl/kubernetes_env.py`). In mock mode `_get_current_replicas()` is
  hard-coded to `(3, 3)` and latency/throughput are sampled from fixed
  per-pattern distributions, so scaling actions have **no effect** on the
  simulated outcome; no HPA/threshold baseline exists in the repo either (its
  "baseline" is static replicas). Controllers therefore cannot be compared inside it.
* **Not attempted:** KIScaler PPO training (needs GPU cluster; out of scope).
* We therefore calibrate our own latency model on KIS-S's real data:
  `latency = base + slope * (EMA(users) / (replicas * cap))^k`, EMA alpha=0.5,
  R^2 = 0.887 on 114 pooled real points (M/M/1 form: 0.72).
  **Replica dependence is an extrapolation** (real data only has replicas = 3).

## Design

| | Static | Drift-triggered (ADWIN) |
|---|---|---|
| Reactive | A: +/-1 threshold rule, anchor calibrated once | D: same; on detection re-derives provisioning anchor |
| Proactive | B: Holt forecaster, bounded params fitted once, state keeps filtering | C: same; on detection re-estimates params on last 15 points |

Regimes: stationary, sudden (30 users->90 at t=30), gradual (ramp t=18..42),
recurring (square wave, half-period 15), cyclic (smooth sinusoid; supplementary).
Pod startup delay = 2 steps; SLO = 3000 ms P95; paired latency noise across arms.

## Headline results (30 seeds, paired, bootstrap 95% CI / Wilcoxon)

Violation rate (lower is better), mean over seeds:

| regime | A | B | C | D |
|---|---|---|---|---|
| sudden | 0.231 | 0.098 | 0.094 | 0.131 |
| gradual | 0.197 | 0.084 | 0.076 | 0.166 |
| recurring | 0.297 | 0.152 | 0.139 | 0.239 |

1. **Adaptation helps the reactive step controller** (D-A violations: -9.9pp sudden,
   -3.1pp gradual, -5.8pp recurring; all p<0.001) **but costs more** (+74, +24, +67 replica-steps).
2. **Adaptation barely helps the proactive controller** (C-B: -0.4 to -1.3pp), because
   a fairly-parameterised static Holt forecaster keeps filtering new data.
3. **Forecasting still wins on SLO:** D does *not* match B/C on violations
   (D-B = +3.3pp sudden, +8.2pp gradual, +8.7pp recurring) though D is cheaper
   (-23 / -65 / -19 replica-steps). So: **a trade-off, not equivalence.**
4. **Hypotheses:** H1 (adaptation effect > family effect) is **not supported**
   on violations; H2 (D matches B/C) is **not supported** on violations,
   supported only in the weak sense of lower cost; H3 not cleanly testable
   (see limitations).

## Sensitivity: how strong is the reactive baseline?

With an **HPA-style proportional** reactive controller, A already gets 12.7% (sudden)
and **D-A = 0.000** (p=1.0, sudden): recalibration has nothing to fix.
Reason: in this simulator drift only shifts load *level*; a proportional controller
is level-invariant and the capacity curve never changes. **The step-rule result in
(1) is therefore partly a property of a slow fixed-step baseline, not of "reactive
controllers" in general.**

## Limitations / deviations (be upfront about all of these)

* **30 seeds instead of the proposal's 5** — with n=5 an exact Wilcoxon test cannot reach p<0.05.
* Seeds only vary latency noise and +/-2-user load jitter (synthetic): real `user_count`
  is a commanded Locust parameter, so the real traces have no natural load noise.
* Controllers are given the calibrated capacity model (as if operators had load-tested).
* Drift is **load-level drift only**. No capacity/service-time drift, which is where
  static calibration would genuinely go stale. Strongest candidate next experiment.
* ADWIN is **blind to the smooth cyclic regime** (never fires; C==B, D==A exactly).
* "Recovery time" is operationalised as longest consecutive SLO-violation run after drift onset.
* The proposal's "threshold calibration by grid search" is simplified to model-based
  anchor derivation for the reactive arms (the Holt arms do use a bounded grid search).
* Latency model R^2 = 0.887 and residual noise sigma = 0.51 (log) on 114 points from
  one deployment; generalisation is untested.
* Bug-fix history worth mentioning in the report: river ADWIN defaults (clock=32)
  never fire in a 60-step episode; unconstrained Holt fit returns alpha=0 on flat
  windows (strawman static arm). Both fixed and documented in `controllers.py`.


---
## Capacity-drift extension (the proposed methodology)

**Why.** The load-level study above showed drift in *offered load* gives
adaptation little to fix (D-A = 0 against a proportional controller). Capacity
drift changes the thing a model-based controller relies on: the useful capacity
of one replica (`capacity_per_replica`, 38.9 users in the calibrated model)
becomes `38.9 * m(t)`. **No real KIS-S measurement contains a capacity change**;
`m(t)` is a synthetic scenario applied to the real-data-calibrated model.

**Regimes** (constant offered load 40 users + jitter; m drops 1.0 -> 0.55):
`cap_stationary` (control), `cap_sudden` (t=20), `cap_gradual` (t=15..40),
`cap_recurring` (alternates every 15 steps).

**Arms** (all see the same load, latency, effective replicas; shared noise per seed):
S = step rule on observed latency (model-free) | RF/PF = proportional / Holt-forecast
controller with capacity belief fixed | RR/PR = same with recalibration |
RO/PO = oracle (true capacity known; upper bound).

**Proposed method (RR, PR).** residual = log(observed latency) - log(model-predicted
latency under believed capacity). A detector watches the residual; on detection the
capacity belief is re-fit (1-D least squares in log-latency) on observations since
the change point, updated each step until 15 observations have accumulated.

**Detector finding (important, deviates from proposal).** ADWIN with the proposal's
delta = 0.002 *never* fires on this residual (0% detection), delta = 0.2 fires in only
27-30% of sudden/gradual episodes (and lag ~20-40 steps). The residual carries
log-noise sd 0.507, so a 0.55-0.6 shift is only ~1.1 sd. A two-sided CUSUM
(k = 0.5, h = 6 in residual-std units; h tuned on no-drift noise only, 4% false alarm
per episode) detects 97-100% with median lag 7-8 (sudden/recurring), 25 (gradual).
All recalibration results below use CUSUM. ADWIN remains the detector in the load-drift study.

**Results (30 seeds, post-drift SLO violation rate; CUSUM for RR/PR)**

| regime | S | RF | RR | RO | PF | PR | PO |
|---|---|---|---|---|---|---|---|
| sudden | 0.130 | 0.388 | 0.172 | 0.092 | 0.268 | 0.125 | 0.068 |
| gradual | 0.104 | 0.281 | 0.167 | 0.081 | 0.187 | 0.099 | 0.039 |
| recurring | 0.122 | 0.274 | 0.179 | 0.096 | 0.191 | 0.118 | 0.066 |

1. **Recalibration works.** PR-PF = -9.5pp / -6.6pp / -5.5pp, RR-RF = -14.3 / -8.5 / -7.1pp
   (sudden/gradual/recurring; all p < 0.001). It costs more replica-steps (+62 to +119).
2. **It does not reach the oracle.** PR-PO = +3.8 / +4.4 / +3.9pp: detection lag and a noisy
   re-fit (belief ~23.5 vs true 21.4 in one example) leave a gap.
3. **Honest caveat: the model-free step rule is competitive.** PR vs S: no significant
   difference in violations (-0.4 to -0.6pp, CIs include 0) while PR costs +14 to +35
   replica-steps more. RR is *worse* than S (+3.9 to +5.1pp). Recalibration fixes the model-based
   controllers' blindness to capacity change, but a controller that reads latency directly
   (like real HPA reading utilisation) never had that blindness. So the claim is: recalibration
   rescues model-based autoscalers under capacity drift, not that it beats feedback control.
4. **No-drift control:** PR-PF = +0.06pp (n.s.); 6.7% of episodes had one false CUSUM alarm.

Outputs: `results/capacity_*.csv`, `figures/capacity_violation.png`, `figures/capacity_belief.png`.
