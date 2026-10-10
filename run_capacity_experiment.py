"""
Capacity-drift experiment: 7 arms x 4 regimes x 30 seeds.
RR / PR are run once per residual detector:
  adwin002 : ADWIN delta=0.002 (the setting used for the load-drift study)
  adwin02  : ADWIN delta=0.2   (more sensitive)
  cusum    : two-sided CUSUM, k=0.5, h=6 (tuned on no-drift noise only)
Outputs results/capacity_results.csv (column 'detector' says which).
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import sys
import pandas as pd
import capacity_drift as cd
from simulator import load_dynamics_params

OUT = os.path.join(BASE, "results")
N_SEEDS = 30
DETECTORS = {"adwin002": ("adwin", 0.002), "adwin02": ("adwin", 0.2), "cusum": ("cusum", None)}


def main(n_seeds=N_SEEDS, cusum_only=False):
    dyn = load_dynamics_params()
    dets = {"cusum": DETECTORS["cusum"]} if cusum_only else DETECTORS
    rows = []
    for regime in cd.CAP_REGIMES:
        for seed in range(n_seeds):
            for arm in ["S", "RF", "RO", "PF", "PO"]:
                r = cd.run_cap_episode(arm, regime, seed, dyn); r["detector"] = "none"; rows.append(r)
            for det, (name, delta) in dets.items():
                cd.DETECTOR["name"] = name
                if delta is not None:
                    cd.DETECTOR["adwin_delta"] = delta
                for arm in ["RR", "PR"]:
                    r = cd.run_cap_episode(arm, regime, seed, dyn); r["detector"] = det; rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/capacity_results.csv", index=False)
    print(len(df), "runs saved")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "cusum-only"]
    main(int(args[0]) if args else N_SEEDS, cusum_only="cusum-only" in sys.argv)
