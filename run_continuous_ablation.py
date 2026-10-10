"""Ablation: change-triggered (CUSUM) re-fit vs. continuous sliding-window re-fit (no detector).
Run after run_capacity_experiment.py. Writes results/continuous_ablation.csv and results/continuous_ablation_contrasts.csv."""
import os, sys, numpy as np, pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import capacity_drift as cd
from scipy.stats import wilcoxon

orig_observe = cd.CapacityBelief.observe
def make_continuous(W):
    def observe(self, t, load, latency, eff):
        a = self.dyn["alpha"]
        self.ema = load if self.ema is None else a * load + (1 - a) * self.ema
        self.obs.append((self.ema, eff, latency))
        if not self.recalibrate:
            return False
        if len(self.obs) >= cd.MIN_FIT_OBS:
            self._refit(self.obs[-W:]); self.n_adaptations += 1
        return False
    return observe

dyn = cd.load_dynamics_params()
rows = []
for regime in cd.CAP_REGIMES:
    for seed in range(30):
        for mode in ["cusum", "cont5", "cont15", "cont30"]:
            if mode == "cusum":
                cd.CapacityBelief.observe = orig_observe; cd.DETECTOR["name"] = "cusum"
            else:
                cd.CapacityBelief.observe = make_continuous(int(mode[4:]))
            for arm in ["RR", "PR"]:
                r = cd.run_cap_episode(arm, regime, seed, dyn); r["mode"] = mode; rows.append(r)
        cd.CapacityBelief.observe = orig_observe
        for arm in ["S", "PF", "RF", "PO", "RO"]:
            r = cd.run_cap_episode(arm, regime, seed, dyn); r["mode"] = "none"; rows.append(r)
df = pd.DataFrame(rows)
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
df.to_csv(os.path.join(BASE, "results", "continuous_ablation.csv"), index=False)
contrast_rows = []
pd.set_option("display.width", 200)
m = "post_drift_violation_rate"
tab = df.assign(lbl=np.where(df["mode"]=="none", df.arm, df.arm+"["+df["mode"]+"]")).groupby(["regime","lbl"]).agg(
    post=(m,"mean"), viol=("slo_violation_rate","mean"), cost=("cost","mean"), caperr=("capacity_rel_error_post","mean")).round(3)
print(tab.to_string())
rng = np.random.default_rng(0)
def get(regime, arm, mode, metric):
    d = df[(df.regime==regime)&(df.arm==arm)&(df["mode"]==mode)].sort_values("seed"); return d[metric].values
print("\nPaired: continuous - cusum (negative = continuous better)")
for regime in cd.CAP_REGIMES:
    for arm in ["PR","RR"]:
        for mode in ["cont5","cont15","cont30"]:
            met = "slo_violation_rate" if regime=="cap_stationary" else m
            diff = get(regime,arm,mode,met) - get(regime,arm,"cusum",met)
            cdiff = get(regime,arm,mode,"cost") - get(regime,arm,"cusum","cost")
            boots=[np.mean(rng.choice(diff,len(diff))) for _ in range(2000)]
            p = 1.0 if np.allclose(diff,0) else wilcoxon(diff).pvalue
            contrast_rows.append(dict(regime=regime, arm=arm, window=mode, metric=met, mean_diff=diff.mean(), ci_lo=np.percentile(boots,2.5), ci_hi=np.percentile(boots,97.5), wilcoxon_p=p, cost_diff=cdiff.mean()))
            print(f"{regime:15s} {arm} {mode:6s} viol {diff.mean():+.4f} CI[{np.percentile(boots,2.5):+.4f},{np.percentile(boots,97.5):+.4f}] p={p:.3g}  cost {cdiff.mean():+.1f}")

pd.DataFrame(contrast_rows).to_csv(os.path.join(BASE, "results", "continuous_ablation_contrasts.csv"), index=False)
