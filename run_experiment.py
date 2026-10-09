"""
Run the full 2x2 factorial experiment.

Main design : arms {A,B,C,D} x regimes {stationary,sudden,gradual,recurring}
              (+ supplementary 'cyclic') x N_SEEDS
Sensitivity : reactive arms {A,D} re-run with an HPA-style proportional
              controller instead of the +/-1 step rule.

N_SEEDS = 30 (proposal said 5): with n=5 paired seeds an exact Wilcoxon
signed-rank test cannot reach p < 0.05 (min two-sided p = 0.0625).
"""
import sys
import pandas as pd
from simulator import run_episode, load_dynamics_params
from drift_regimes import REGIMES
from config import RESULTS_DIR as OUT

N_SEEDS = 30
ARMS = ["A", "B", "C", "D"]


def main(n_seeds=N_SEEDS):
    dyn = load_dynamics_params()
    rows = []
    for regime in REGIMES:
        for seed in range(n_seeds):
            for arm in ARMS:
                rows.append(run_episode(arm, regime, seed, dyn))
    main_df = pd.DataFrame(rows)
    main_df.to_csv(f"{OUT}/main_results.csv", index=False)

    rows = []
    for regime in REGIMES:
        for seed in range(n_seeds):
            for arm in ["A", "D"]:
                rows.append(run_episode(arm, regime, seed, dyn, reactive_mode="proportional"))
    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(f"{OUT}/sensitivity_hpa_results.csv", index=False)
    print(f"main: {len(main_df)} runs, sensitivity: {len(sens_df)} runs saved to {OUT}/")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SEEDS)
