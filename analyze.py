"""
Analysis of the 2x2 factorial results.

Paired (by seed) contrasts per regime, since every arm sees identical load
trace and latency-noise sequence for a given (regime, seed):
  adaptation effect, reactive   : D - A
  adaptation effect, proactive  : C - B
  family effect, static         : B - A
  family effect, adaptive       : C - D
  interaction                   : (C - B) - (D - A)
  headline (RQ)                 : D - B , D - C   (recalibrated threshold vs forecast)
Bootstrap 95% CI on the mean paired difference + Wilcoxon signed-rank p.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR as RES, FIGURES_DIR as FIG

METRICS = ["cost", "slo_violation_rate"]
rng = np.random.default_rng(0)


def paired(df, regime, metric, plus, minus):
    d = df[df.regime == regime]
    a = d[d.arm == plus].sort_values("seed")[metric].values
    b = d[d.arm == minus].sort_values("seed")[metric].values
    return a - b


def summarize_diff(diff):
    mean = float(np.mean(diff))
    boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if np.allclose(diff, 0):
        p = 1.0
    else:
        try:
            p = float(wilcoxon(diff).pvalue)
        except ValueError:
            p = 1.0
    return mean, float(lo), float(hi), p


def contrasts_table(df, regimes):
    defs = [("adapt | reactive (D-A)", "D", "A"), ("adapt | proactive (C-B)", "C", "B"),
            ("family | static (B-A)", "B", "A"), ("family | adaptive (C-D)", "C", "D"),
            ("HEADLINE D-B", "D", "B"), ("HEADLINE D-C", "D", "C")]
    rows = []
    for regime in regimes:
        for metric in METRICS:
            for name, p_, m_ in defs:
                mean, lo, hi, p = summarize_diff(paired(df, regime, metric, p_, m_))
                rows.append({"regime": regime, "metric": metric, "contrast": name,
                             "mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p})
            inter = paired(df, regime, metric, "C", "B") - paired(df, regime, metric, "D", "A")
            mean, lo, hi, p = summarize_diff(inter)
            rows.append({"regime": regime, "metric": metric, "contrast": "interaction (C-B)-(D-A)",
                         "mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(f"{RES}/main_results.csv")
    sens = pd.read_csv(f"{RES}/sensitivity_hpa_results.csv")
    regimes = ["stationary", "sudden", "gradual", "recurring", "cyclic"]

    agg = df.groupby(["regime", "arm"]).agg(
        cost=("cost", "mean"), cost_sd=("cost", "std"),
        viol=("slo_violation_rate", "mean"), viol_sd=("slo_violation_rate", "std"),
        post_viol=("post_drift_violation_rate", "mean"),
        longest_run=("longest_violation_run", "mean"),
        churn=("churn", "mean"), adapt=("n_adaptations", "mean"),
        det_lag=("detection_lag", "mean"), false_det=("false_detections", "mean"),
    ).round(3)
    agg = agg.reindex(regimes, level=0)
    agg.to_csv(f"{RES}/summary_by_regime_arm.csv")
    print("=== MEAN METRICS (30 seeds) ===")
    print(agg.to_string())

    ct = contrasts_table(df, regimes)
    ct.to_csv(f"{RES}/paired_contrasts.csv", index=False)

    sens_agg = sens.groupby(["regime", "arm"]).agg(
        cost=("cost", "mean"), viol=("slo_violation_rate", "mean"),
        adapt=("n_adaptations", "mean")).round(3).reindex(regimes, level=0)
    sens_agg.to_csv(f"{RES}/sensitivity_hpa_summary.csv")
    print("\n=== SENSITIVITY: HPA-style proportional reactive (A,D) ===")
    print(sens_agg.to_string())

    sens_ct = contrasts_table(
        pd.concat([sens.assign(arm=sens.arm.map({"A": "A", "D": "D"})),
                   df[df.arm.isin(["B", "C"])]]), regimes)
    sens_ct.to_csv(f"{RES}/sensitivity_hpa_contrasts.csv", index=False)

    # ---- figure: cost vs violation per regime ----
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), sharey=False)
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
    for ax, regime in zip(axes, regimes):
        sub = agg.loc[regime]
        for arm in "ABCD":
            ax.errorbar(sub.loc[arm, "cost"], sub.loc[arm, "viol"],
                        xerr=sub.loc[arm, "cost_sd"] / np.sqrt(30) * 1.96,
                        yerr=sub.loc[arm, "viol_sd"] / np.sqrt(30) * 1.96,
                        fmt="o", color=colors[arm], label=arm, capsize=3)
        ax.set_title(regime); ax.set_xlabel("cost (replica-steps)")
        ax.set_ylabel("SLO violation rate"); ax.grid(alpha=0.3)
    axes[0].legend(title="arm (mean, 95% CI)")
    plt.tight_layout(); plt.savefig(f"{FIG}/cost_vs_violation.png", dpi=130)

    # ---- figure: example trajectories (sudden, seed 0) ----
    from simulator import run_episode
    fig, axes = plt.subplots(2, 4, figsize=(18, 6), sharex=True)
    for j, arm in enumerate("ABCD"):
        r = run_episode(arm, "sudden", 0, keep_records=True)["records"]
        axes[0, j].plot(r["load"], color="gray"); axes[0, j].set_title(f"arm {arm}: load / replicas")
        ax2 = axes[0, j].twinx(); ax2.step(range(60), r["replicas"], color=colors[arm], where="post")
        axes[1, j].plot(r["latency"], color=colors[arm]); axes[1, j].axhline(3000, color="k", ls="--", lw=1)
        axes[1, j].set_title("latency (ms) vs SLO"); axes[1, j].set_xlabel("step")
    plt.tight_layout(); plt.savefig(f"{FIG}/sudden_trajectories.png", dpi=130)
    print("\nfigures saved")


if __name__ == "__main__":
    main()
