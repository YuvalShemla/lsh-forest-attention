#!/usr/bin/env python3
"""
Run all claude_experiments sequentially.
Prints progress logs for each experiment.
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

EXPERIMENTS = [
    ("01_per_layer_diagnostics", "Experiment 1: Per-Layer Diagnostics"),
    ("02_crossover_budget", "Experiment 2: Crossover Budget Verification"),
    ("03_nearest_value_key", "Experiment 3: Nearest-Value Key Selection (A')"),
    ("04_kmeans_vs_gmm", "Experiment 4: K-Means vs GMM Partition"),
    ("05_gmm_bias_ablation", "Experiment 5: GMM Bias Source Ablation"),
]


def run_experiment(folder, title):
    exp_dir = SCRIPT_DIR / folder
    script = exp_dir / "experiment.py"

    if not script.exists():
        print(f"  SKIP: {script} not found")
        return False

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"  Directory: {exp_dir}")
    print(f"{'=' * 70}\n", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        cwd=str(exp_dir),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  DONE: {title} ({elapsed:.0f}s / {elapsed/60:.1f}min)")
    else:
        print(f"\n  FAILED: {title} (exit code {result.returncode}, {elapsed:.0f}s)")

    return result.returncode == 0


def main():
    print("=" * 70)
    print("  RUNNING ALL CLAUDE EXPERIMENTS")
    print(f"  Python: {sys.executable}")
    print(f"  Root:   {SCRIPT_DIR}")
    print("=" * 70)

    total_start = time.time()
    results = {}

    for folder, title in EXPERIMENTS:
        success = run_experiment(folder, title)
        results[title] = success

    total_time = time.time() - total_start

    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for title, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status:>6s}] {title}")
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
