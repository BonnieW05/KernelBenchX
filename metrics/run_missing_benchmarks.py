"""
Scan `golden_metrics/*_perf.py` and generate (or overwrite) the corresponding
JSON timing results in the "golden reference" directory.

The golden reference directory is resolved by `golden_path.resolve_golden_results_dir()`:
  - export KERNELBENCHX_GOLDEN_RESULTS_DIR=/path/to/dir
  - or export KERNELBENCHX_GOLDEN_MACHINE=4090  → .../golden_results/4090/
  - or run: python run_missing_benchmarks.py --machine 4090

By default this script only runs `TARGET_BENCHMARKS`. Use `--all` for a full scan.
Use `--missing-only` to fill in missing JSONs only.

Examples:
  cd metrics && python run_missing_benchmarks.py --machine 4090 --all
  export KERNELBENCHX_GOLDEN_MACHINE=a100 && python run_missing_benchmarks.py --all

The script sets `KERNELBENCHX_RESULTS_PATH` for subprocesses to the resolved directory.
Timeout is controlled by `KERNELBENCHX_BENCH_TIMEOUT` (seconds), default: 180.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_KBX_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_KBX_ROOT, "EVAL") not in sys.path:
    sys.path.insert(0, os.path.join(_KBX_ROOT, "EVAL"))
from golden_path import resolve_golden_results_dir

# Default: only rerun these targets (vs `--all`); adjust as needed.
TARGET_BENCHMARKS: tuple[str, ...] = (
    "adaptive_avg_pool2d",
    "fused_hstack_div",
    "solve_and_add_scaled_vector",
    "tril_mm_and_scale",
)


def _subprocess_diag(result: subprocess.CompletedProcess) -> str:
    chunks: list[str] = []
    if result.stderr and result.stderr.strip():
        chunks.append(result.stderr.rstrip())
    if result.stdout and result.stdout.strip():
        chunks.append("--- stdout ---\n" + result.stdout.rstrip())
    return "\n".join(chunks) if chunks else "(no stderr/stdout)"


def _discover_benchmarks(golden_metrics_dir: str) -> list[str]:
    names: list[str] = []
    for fn in sorted(os.listdir(golden_metrics_dir)):
        if fn.endswith("_perf.py"):
            names.append(fn[: -len("_perf.py")])
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="Run golden *_perf.py into golden_results[/machine]/")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run all *_perf.py under golden_metrics (default: only TARGET_BENCHMARKS).",
    )
    ap.add_argument(
        "--missing-only",
        action="store_true",
        help="Only run benchmarks whose .json does not exist in the target directory.",
    )
    ap.add_argument(
        "--machine",
        "-m",
        default=None,
        metavar="NAME",
        help="Write into golden_results/<NAME>/; sets KERNELBENCHX_GOLDEN_MACHINE unless GOLDEN_RESULTS_DIR is set.",
    )
    args = ap.parse_args()

    if args.machine is not None:
        if os.environ.get("KERNELBENCHX_GOLDEN_RESULTS_DIR", "").strip():
            print(
                "KERNELBENCHX_GOLDEN_RESULTS_DIR is set; ignoring --machine.",
                file=sys.stderr,
            )
        else:
            os.environ["KERNELBENCHX_GOLDEN_MACHINE"] = args.machine.strip()

    base_dir = os.path.abspath(os.path.dirname(__file__))
    golden_metrics_dir = os.path.join(base_dir, "golden_metrics")
    golden_results_dir = resolve_golden_results_dir()
    kbx_root = _KBX_ROOT
    data_dir = os.path.join(kbx_root, "data")

    os.makedirs(golden_results_dir, exist_ok=True)

    if args.all:
        benchmarks = _discover_benchmarks(golden_metrics_dir)
    else:
        benchmarks = list(TARGET_BENCHMARKS)

    if args.missing_only:
        benchmarks = [
            b
            for b in benchmarks
            if not os.path.isfile(os.path.join(golden_results_dir, f"{b}.json"))
        ]

    timeout_s = int(os.environ.get("KERNELBENCHX_BENCH_TIMEOUT", "180"))

    py = sys.executable
    env = os.environ.copy()
    env["KERNELBENCHX_RESULTS_PATH"] = golden_results_dir
    pp = data_dir
    if env.get("PYTHONPATH"):
        pp = data_dir + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pp

    print(f"Running {len(benchmarks)} benchmark(s) -> {golden_results_dir}")
    print(f"Python (subprocesses use the same interpreter): {py}")
    print("=" * 60)

    failed: list[tuple[str, str]] = []
    succeeded: list[str] = []

    for benchmark in benchmarks:
        perf_file = os.path.join(golden_metrics_dir, f"{benchmark}_perf.py")
        result_file = os.path.join(golden_results_dir, f"{benchmark}.json")

        if not os.path.exists(perf_file):
            print(f"skip {benchmark}: missing {perf_file}")
            failed.append((benchmark, "perf file not found"))
            continue

        print(f"\nRunning {benchmark}...")
        try:
            result = subprocess.run(
                [py, perf_file],
                cwd=golden_metrics_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )

            if result.returncode == 0:
                if os.path.exists(result_file):
                    print(f"OK {benchmark}")
                    succeeded.append(benchmark)
                else:
                    print(f"WARN {benchmark}: exit 0 but no {result_file}")
                    failed.append((benchmark, "no result file"))
            else:
                diag = _subprocess_diag(result)
                print(f"FAIL {benchmark} (exit {result.returncode})\n{diag}\n")
                failed.append((benchmark, diag))
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT {benchmark} ({timeout_s}s)")
            failed.append((benchmark, "timeout"))
        except Exception as e:
            print(f"EXC {benchmark}: {e}")
            failed.append((benchmark, str(e)))

    print("\n" + "=" * 60)
    print(f"Summary: ok {len(succeeded)}/{len(benchmarks)}, fail {len(failed)}/{len(benchmarks)}")

    if succeeded:
        print("Succeeded:", ", ".join(succeeded[:20]) + (" ..." if len(succeeded) > 20 else ""))

    if failed:
        print("Failed (full output is shown in the FAIL blocks above; repeated here for easier searching):")
        for b, reason in failed:
            print(f"\n--- {b} ---\n{reason}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
