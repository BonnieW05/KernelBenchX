#!/usr/bin/env python3
"""Scan unified metrics.json files under experiment result trees.

Example:
  python scripts/analyze_experiment_metrics.py \\
    --base /path/to/experiment_results \\
    --agent geak \\
    --gbs-threshold 4000

By default this script reports:
  - tasks with exe_ok=True but perf_ok=False (passed correctness but failed perf)
  - tasks with extremely high GB/s (often indicates a mismatch in the byte-count
    definition in the corresponding *_perf.py template)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Analyze unified metrics.json under experiment result trees.")
    ap.add_argument(
        "--base",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "experiment_results"),
        help="Experiment root (contains GPU subfolders such as 4090/5090/a100/...).",
    )
    ap.add_argument("--agent", type=str, default="geak", help="Subdirectory name, e.g., geak or kernelagent.")
    ap.add_argument("--gbs-threshold", type=float, default=4000.0, help="Flag tasks whose GB/s exceeds this threshold.")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    if not os.path.isdir(base):
        print(f"Directory does not exist: {base}")
        return 1

    exe_not_perf: dict[str, list[str]] = defaultdict(list)
    high_gbs: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for gpu in sorted(os.listdir(base)):
        mp = os.path.join(base, gpu, args.agent, "metrics.json")
        if not os.path.isfile(mp):
            continue
        data = load_metrics(mp)
        for key, row in data.items():
            ex = row.get("exe_ok")
            pr = row.get("perf_ok")
            if ex and not pr:
                exe_not_perf[gpu].append(key)
            gbs = row.get("gbs")
            if isinstance(gbs, (int, float)) and gbs >= args.gbs_threshold:
                high_gbs[gpu].append((key, float(gbs)))

    print("=" * 72)
    print(f"base={base}  agent={args.agent}")
    print("=" * 72)

    print("\n[1] exe_ok=True and perf_ok=False (passed correctness, failed perf)")
    any_a = False
    for gpu in sorted(exe_not_perf.keys()):
        any_a = True
        print(f"\n  --- {gpu} ({len(exe_not_perf[gpu])}) ---")
        for name in exe_not_perf[gpu]:
            print(f"    {name}")
    if not any_a:
        print("  (none)")

    print(
        f"\n[2] gbs >= {args.gbs_threshold} (suspiciously high; check byte-count / FLOPs definitions in *_perf.py)"
    )
    any_b = False
    for gpu in sorted(high_gbs.keys()):
        any_b = True
        print(f"\n  --- {gpu} ---")
        for name, g in sorted(high_gbs[gpu], key=lambda x: -x[1]):
            print(f"    {g:12.2f}  {name}")
    if not any_b:
        print("  (none)")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
