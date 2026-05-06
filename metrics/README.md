# Performance metrics (golden reference timing)

This directory contains:
- **`golden_metrics/*_perf.py`**: per-operator timing templates that benchmark the reference implementations
- **`golden_results/<machine>/*.json`**: measured timing results (per input size: `ms`, `GB/s`, `TFLOPS`)

The golden JSONs are consumed by `EVAL/2_efficiency.py` to compute speedups by comparing:
- generated kernels' `perf_results/*`
- vs the golden reference in `golden_results/<machine>/*.json`

## Per-GPU golden references
Golden results are hardware-specific.
- Do not compare results across different GPUs
- Use the environment variables defined in `EVAL/golden_path.py` to select a subfolder under `golden_results/` (e.g., `golden_results/4090/`)

## Regenerating / updating golden JSONs
Run:
```bash
python run_missing_benchmarks.py --help
```
Supported options:

* --machine
* --all
* --missing-only

Timeout is controlled by KERNELBENCHX_BENCH_TIMEOUT.