# KernelBench_X

[KernelBench\_X](https://arxiv.org/abs/2605.04956) is a reproducible evaluation harness for **Triton kernel code generation**. It measures:

- **Buildability (Call)**: Can the submitted code compile and run?
- **Numerical correctness (Exe)**: Does it match a reference implementation under a deterministic test suite?
- **Efficiency (Perf)**: runtime (ms), throughput (TFLOPS), memory bandwidth (GB/s), and speedup vs. a GPU-matched golden reference.
- **Others**: lightweight code quality signals (e.g., maintainability index via `radon`) are also reported.

**Paper:** https://arxiv.org/pdf/2605.04956

## Acknowledgment

KernelBench_X builds upon prior efforts in Triton kernel benchmarking, particularly [TritonBench](https://github.com/thunlp/TritonBench), while introducing:
- stricter correctness verification,
- extended task coverage (e.g., quantization, multi-precision),
- and a unified evaluation pipeline for correctness and efficiency analysis.

We thank the authors of these projects for providing foundational resources.

## Setup & Run

> **Recommended (no sudo):** `conda create -n kbx python=3.11 -y && conda activate kbx`  
> System Python users may need: `sudo apt install python3.10-dev`

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/data:${PYTHONPATH}"

# First argument is passed to EVAL/0_call_acc.py --source (see Input format).
bash scripts/run_eval.sh [--bench-timeout SEC] <source> <output_dir> <gpus>

# Equivalent wrapper when your artifacts are already .py files or a folder (same flags).
bash scripts/run_eval_from_files.sh [--bench-timeout SEC] <source> <output_dir> <gpus>
```

Examples:

```bash
bash scripts/run_eval.sh predictions.jsonl ./out_run 0
bash scripts/run_eval_from_files.sh ./generated/attention.py ./out_run 0
bash scripts/run_eval_from_files.sh ./generated/kernels/ ./out_run 0,1
```

Outputs land in `./out_run/`: `metrics.json` (per-task results), `summary.json` (aggregate pass rates and speedup), and `intermediate/` (per-stage JSONLs, perf scripts, raw logs).

## Quickstart notebook

Run `examples/Examples.ipynb` to verify your environment and understand the full pipeline (**call → exe → perf**) end-to-end.

It demonstrates three submission formats: JSONL, Single, Directory.


## Input format

The `--source` path may be:

1. **JSONL file** — one JSON object per line. Markdown fences in `predict` are stripped automatically.
   ```jsonc
   { "predict": "<generated code>", "file": "Fusion/attention.py" }
   ```
   `file` must be provided on **every** line (recommended).
   If you omit `file`, each JSONL line must instead include `instruction`, which must contain:
   `Functional Description: ` and `Wrapper Entry Information:`
   ```jsonc
   { "instruction": "Functional Description: <...>\nWrapper Entry Information: <...>", "predict": "<generated code>" }
   ```
  
2. **Single `.py` file** — the **basename** must match a task in `data/kernelbenchx/` (e.g. `attention.py`). If the file contains the dataset separator line (`#` repeated 146 times), only the content **above** that line is treated as the submission.

3. **Directory** — recursively collects `*.py` and `*.jsonl`. Each file is evaluated as in (1) or (2); every basename must map to a benchmark task.

**Kernel entry contract.** To ensure the submission is executable, define a **top-level** callable named either `kernel_function` or the target task stem (for example, `attention` for `attention.py`).


## GPU-Specific Golden Timing Baselines

Golden timing JSONs are GPU-specific, stored under `metrics/golden_results/<machine>/`. Resolution precedence: `KERNELBENCHX_GOLDEN_RESULTS_DIR` (explicit override) → `KERNELBENCHX_GOLDEN_MACHINE` (named subfolder) → default `5090`.

```bash
export KERNELBENCHX_GOLDEN_MACHINE=a100
bash scripts/run_eval.sh predictions.jsonl ./out_run 0
```

To regenerate golden timings on a given machine:

```bash
cd metrics && python run_missing_benchmarks.py --machine 4090 --all
```

## Repository layout

```
KernelBench_X/
├── data/                     # task metadata + reference implementations + iteration corpus
├── EVAL/                     # pipeline stages (call → exe → perf)
├── metrics/                  # golden references + perf scripts
├── scripts/                  # user-facing entrypoints
└── utils/                    # shared helpers
```

## Citation
**If you use this code or find our work valuable, please cite:**
```
@article{wang2026kernelbenchx,
  title={KernelBench-X: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels}, 
  author={Wang, Han and Zhang, Jintao and Jiang, Kai and Wang, Haoxu and Chen, Jianfei and Zhu, Jun},
  journal={arXiv preprint arXiv:2605.04956},
  year={2026}
}
```
