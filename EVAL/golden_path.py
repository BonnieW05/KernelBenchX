"""
Resolve the directory that contains golden reference perf JSONs (supports per-machine subfolders).

Precedence (highest to lowest):
1. KERNELBENCHX_GOLDEN_RESULTS_DIR — any directory (absolute or relative), used as-is.
2. KERNELBENCHX_GOLDEN_MACHINE — a single path segment; resolves to golden_results/<name>/.
3. If MACHINE is not set: use KERNELBENCHX_DEFAULT_GOLDEN_MACHINE (default: **5090**), i.e.
   metrics/golden_results/5090/

If the repo stores golden references under golden_results/<gpu>/ with no JSONs at the root,
you must use (2) or rely on the default (5090). If you still use a flat layout
golden_results/*.json, set KERNELBENCHX_GOLDEN_RESULTS_DIR to that directory.
"""
from __future__ import annotations

import os
import re

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PERF_DIR = os.path.normpath(os.path.join(_EVAL_DIR, "../metrics"))


def resolve_golden_results_dir() -> str:
    explicit = os.environ.get("KERNELBENCHX_GOLDEN_RESULTS_DIR", "").strip()
    if explicit:
        return os.path.normpath(os.path.expanduser(explicit))

    base = os.path.join(_PERF_DIR, "golden_results")
    sub = os.environ.get("KERNELBENCHX_GOLDEN_MACHINE", "").strip()
    if not sub:
        # Default to 5090 when not specified (matches this repo's golden_results/<gpu>/ layout).
        sub = os.environ.get("KERNELBENCHX_DEFAULT_GOLDEN_MACHINE", "5090").strip()
    if not sub:
        return os.path.normpath(base)

    if not re.match(r"^[A-Za-z0-9_.-]+$", sub):
        raise ValueError(
            "KERNELBENCHX_GOLDEN_MACHINE / DEFAULT must be a single path segment "
            "(letters/digits/_/./-). '/' and '..' are not allowed. "
            f"Got: {sub!r}"
        )
    return os.path.normpath(os.path.join(base, sub))
