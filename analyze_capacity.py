"""
Analysis of the capacity-drift experiment (paired by seed).
Contrasts (negative violation difference = first arm better):
  RR-RF, PR-PF : effect of recalibration (the proposed method)
  RO-RR, PO-PR : gap left to the oracle (how much more could be gained)
  RR-S, PR-S   : model-based + recalibration vs model-free feedback rule
  PR-RR        : proactive vs reactive, both recalibrated
Each recalibrated arm is evaluated once per detector.
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
rng = np.random.default_rng(0)
DETS = ["adwin002", "adwin02", "cusum"]   # filtered to detectors present in the results file
REGS = ["cap_stationary", "cap_sudden", "cap_gradual", "cap_recurring"]


def get(df, regime, arm, metric, det=None):
    d = df[(df.regime == regime) & (df.arm == arm)]
    if det and arm in ("RR", "PR"):
        d = d[d.detector == det]
    return d.sort_values("seed")[metric].values


def summ(diff):
    mean = float(np.mean(diff))
    boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff).pvalue)
    return mean, float(lo), float(hi), p


def main():
    df = pd.read_csv(f"{RES}/capacity_results.csv")
    global DETS
    DETS = [d for d in DETS if d in set(df.detector)]

    # 1. per regime / arm summary (recalibrated arms split by detector)
    df["arm_label"] = np.where(df.detector.isin(["none"]), df.arm, df.arm + "[" + df.detector + "]")
    s = (df.groupby(["regime", "arm_label"])
           .agg(violation=("slo_violation_rate", "mean"), post_violation=("post_drift_violation_rate", "mean"),
                cost=("cost", "mean"), detections=("n_detections", "mean"),
                false_det=("false_detections", "mean"), detect_lag=("detection_lag", "mean"),
                cap_err=("capacity_rel_error_post", "mean")).round(3).reset_index())
    s.to_csv(f"{RES}/capacity_summary.csv", index=False)

    # 2. paired contrasts
    defs = [("recal effect | reactive (RR-RF)", "RR", "RF", True), ("recal effect | proactive (PR-PF)", "PR", "PF", True),
            ("gap to oracle | reactive (RR-RO)", "RR", "RO", True), ("gap to oracle | proactive (PR-PO)", "PR", "PO", True),
            ("RR vs model-free step rule (RR-S)", "RR", "S", True), ("PR vs model-free step rule (PR-S)", "PR", "S", True),
            ("proactive vs reactive, recal (PR-RR)", "PR", "RR", True),
            ("fixed proactive vs step rule (PF-S)", "PF", "S", False)]
    rows = []
    for regime in REGS:
        for metric in ["slo_violation_rate", "cost"]:
            for name, a, b, det_dep in defs:
                for det in (DETS if det_dep else [None]):
                    diff = get(df, regime, a, metric, det) - get(df, regime, b, metric, det)
                    mean, lo, hi, p = summ(diff)
                    rows.append({"regime": regime, "metric": metric, "contrast": name, "detector": det or "-",
                                 "mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p})
    c = pd.DataFrame(rows).round(4)
    c.to_csv(f"{RES}/capacity_contrasts.csv", index=False)

    # 2b. the same contrasts on the POST-DRIFT violation rate (the metric reported in the paper's
    #     Part 2 table and text). Separate RNG so the bootstrap CIs in capacity_contrasts.csv are unchanged.
    global rng
    rng = np.random.default_rng(1)
    prow = []
    for regime in [r for r in REGS if r != "cap_stationary"]:
        for name, a_, b_, det_dep in defs:
            for det in (["cusum"] if det_dep else [None]):
                diff = get(df, regime, a_, "post_drift_violation_rate", det) - get(df, regime, b_, "post_drift_violation_rate", det)
                cdiff = get(df, regime, a_, "cost", det) - get(df, regime, b_, "cost", det)
                mean, lo, hi, p = summ(diff)
                prow.append({"regime": regime, "metric": "post_drift_violation_rate", "contrast": name,
                             "detector": det or "-", "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                             "wilcoxon_p": p, "cost_diff": float(np.mean(cdiff))})
    pd.DataFrame(prow).round(4).to_csv(f"{RES}/capacity_post_contrasts.csv", index=False)

    # 3. detector quality on the recalibrated arms
    det = (df[df.arm == "RR"].groupby(["regime", "detector"])
             .agg(detect_rate=("detection_lag", lambda x: x.notna().mean()), median_lag=("detection_lag", "median"),
                  false_alarms_per_episode=("false_detections", "mean")).round(3).reset_index())
    det.to_csv(f"{RES}/capacity_detector_quality.csv", index=False)

    # 4. figure: post-drift violation by arm (cusum for recal)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    order = ["S", "RF", "RR", "RO", "PF", "PR", "PO"]
    for ax, regime in zip(axes, REGS[1:]):
        vals = [get(df, regime, a, "post_drift_violation_rate", "cusum").mean() for a in order]
        errs = [1.96 * get(df, regime, a, "post_drift_violation_rate", "cusum").std(ddof=1) / np.sqrt(30) for a in order]
        cols = ["#888", "#c44", "#2a7", "#222", "#e93", "#27a", "#222"]
        ax.bar(order, vals, yerr=errs, color=cols, capsize=3)
        ax.set_title(regime); ax.set_ylabel("post-drift SLO violation rate")
    plt.suptitle("Capacity drift: violation after onset (recalibrated arms use CUSUM; bars = mean, 95% CI over 30 seeds)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{FIG}/capacity_violation.png", dpi=130); plt.close()

    # 5. figure: one trajectory of the capacity belief (cap_sudden, seed 0)
    import capacity_drift as cd
    cd.DETECTOR["name"] = "cusum"
    fig, ax = plt.subplots(figsize=(7, 3.8))
    r = cd.run_cap_episode("PR", "cap_sudden", 0, keep_records=True)["records"]
    ax.plot(r["true_cap"], "k-", label="true capacity / replica")
    ax.plot(r["cap_hat"], "g--", label="belief, recalibrated (PR)")
    r2 = cd.run_cap_episode("PF", "cap_sudden", 0, keep_records=True)["records"]
    ax.plot(r2["cap_hat"], "r:", label="belief, fixed (PF)")
    ax.set_xlabel("step"); ax.set_ylabel("users / replica"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/capacity_belief.png", dpi=130); plt.close()

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(s[s.regime != "cap_stationary"].to_string(index=False))
    print(); print(det.to_string(index=False))
    print(); print(c[(c.metric == "slo_violation_rate") & (c.detector.isin(["cusum", "-"]))].to_string(index=False))


if __name__ == "__main__":
    main()
