"""
Extract real request-rate / P95 latency time series from KIS-S's released
sample experimental data (results/cpu_baseline/dynamic/*_stats.json).

These are genuine Locust measurements recorded against a real Kubernetes +
Triton inference deployment (mobilenetv4), collected under a fixed replica
count (no autoscaling active during collection). We use them as the
ground-truth traffic + latency-under-fixed-capacity signal to calibrate our
own dynamics model, since KIS-S's released mock-mode RL environment has no
action-outcome coupling (see project notes).
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
import json
import glob
import os
import pandas as pd

KISIM_RESULTS = os.path.join(os.environ.get("KISIM_DIR", os.path.join(BASE, "..", "KISim")), "results", "cpu_baseline", "dynamic")
OUT_DIR = os.path.join(BASE, "data")

PATTERNS = ["ramp", "spike", "periodic", "random"]


def extract_one(stats_path, users_path):
    with open(stats_path) as f:
        snapshots = json.load(f)
    user_counts = None
    if os.path.exists(users_path):
        with open(users_path) as f:
            user_counts = json.load(f)["user_counts"]

    rows = []
    for i, snap in enumerate(snapshots):
        agg = next((s for s in snap.get("stats", []) if s["name"] == "Aggregated"), None)
        if agg is None:
            continue
        # "current_response_time_percentiles" (top level) is the windowed /
        # recent value; the per-endpoint "response_time_percentile_0.95"
        # inside stats[] is cumulative-since-test-start and monotonically
        # increasing by construction, so we deliberately use the former.
        cur_pct = snap.get("current_response_time_percentiles", {}) or {}
        rows.append({
            "step": i,
            "request_rate": agg.get("current_rps", 0.0),
            "p95_latency_ms": cur_pct.get("response_time_percentile_0.95", None),
            "p50_latency_ms": cur_pct.get("response_time_percentile_0.5", None),
            "num_requests_cum": agg.get("num_requests", 0),
            "fail_ratio": snap.get("fail_ratio", 0.0),
            "user_count": (user_counts[i] if user_counts and i < len(user_counts) else None),
        })
    df = pd.DataFrame(rows)
    # Fill any missing p95 (first snapshot sometimes has no window yet) by
    # back-filling from the next valid value.
    df["p95_latency_ms"] = df["p95_latency_ms"].bfill().ffill()
    df["user_count"] = df["user_count"].ffill().bfill()
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for pattern in PATTERNS:
        stats_matches = sorted(glob.glob(os.path.join(KISIM_RESULTS, f"{pattern}_*_stats.json")))
        if not stats_matches:
            print(f"WARNING: no file found for pattern={pattern}")
            continue
        stats_path = stats_matches[0]
        users_path = stats_path.replace("_stats.json", "_users.json")
        df = extract_one(stats_path, users_path)
        out_path = os.path.join(OUT_DIR, f"{pattern}.csv")
        df.to_csv(out_path, index=False)
        summary.append(pattern)
        print(f"{pattern:10s} <- {os.path.basename(stats_path)}  ({len(df)} steps)  "
              f"rps[{df['request_rate'].min():.1f},{df['request_rate'].max():.1f}]  "
              f"users[{df['user_count'].min():.0f},{df['user_count'].max():.0f}]  "
              f"p95[{df['p95_latency_ms'].min():.0f},{df['p95_latency_ms'].max():.0f}]ms")

    print(f"\nSaved {len(summary)} traces to {OUT_DIR}/")


if __name__ == "__main__":
    main()
