#!/usr/bin/env python3

import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import zip_longest

BASE_DIR = Path(os.environ.get("KERNELBENCHX_COLLECT_DIR", Path(__file__).parent / "collected_data"))
_ERROR_STATS_FILE = BASE_DIR / "error_stats.json"

def _load_seen_errors() -> defaultdict:
    if _ERROR_STATS_FILE.exists():
        with _ERROR_STATS_FILE.open('r', encoding='utf-8') as f:
            data = json.load(f)
            return defaultdict(int, {tuple(k.split('|')): v for k, v in data.items()})
    return defaultdict(int)

def _save_seen_errors():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    data = {'|'.join(k): v for k, v in _seen_errors.items()}
    with _ERROR_STATS_FILE.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

_seen_errors = _load_seen_errors()

ERROR_PATTERNS = [
    ('oom_error', {'out of memory', 'cuda out of memory', 'oom', 'memoryerror', 'alloc'}),
    ('timeout_error', {'time out', 'timed out', 'deadline exceeded'}),
    
    ('syntax_error', {'syntaxerror', 'indentationerror', 'taberror', 'invalid syntax', 'unexpected indent'}),
    ('import_error', {'importerror', 'modulenotfounderror', 'no module named', 'cannot import'}),
    
    ('shape_mismatch', {'shape', 'dimension', 'broadcast', 'size mismatch', 'mat1 and mat2', 'match the size'}),
    ('device_error', {'expected all tensors', 'device', 'cuda', 'cpu', 'gpu', 'found at least two devices'}),
    ('dtype_error', {'expected scalar type', 'found type', 'dtype', 'input type'}),

    ('index_error', {'indexerror', 'list index', 'out of bounds', 'out of range'}),
    ('key_error', {'keyerror', 'not in index'}),
    ('attribute_error', {'attributeerror', 'has no attribute', 'object has no'}),

    ('assertion_error', {'assertionerror', 'assert'}),
    ('type_error', {'typeerror', 'argument', 'positional argument', 'keyword argument'}),
    
    ('value_error', {'valueerror', 'invalid value', 'must be'}),
    ('runtime_error', {'runtimeerror', 'execution failed', 'traceback'}),
]

def classify_error(msg: str) -> str:
    if not msg: return 'unknown'
    m = msg.lower()
    return next((cat for cat, pats in ERROR_PATTERNS if any(p in m for p in pats)), 'other')

def _append_jsonl(filename: str, data: dict):
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASE_DIR / f"{filename}.jsonl"
    data['ts'] = datetime.now().isoformat()
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

SPEEDUP_THRESHOLD = 3.0

def collect_good_impl(file_name: str, code: str, speedup: float, model: str = ""):
    if speedup < SPEEDUP_THRESHOLD:
        return
    _append_jsonl("good_implementations", {
        "file": file_name, "code": code, "speedup": speedup, "model": model
    })

def _build_diff_record(file_name: str, before: str, after: str, model: str = ""):
    similarity = SequenceMatcher(None, before, after).ratio()
    is_large = similarity < 0.6 or len(after) > 70
    
    record = {"file": file_name, "model": model, "is_large_change": is_large}
    if is_large:
        record.update({"before": before, "after": after})
    else:
        record["after"] = after
        record["diff_lines"] = [
            {"line": i+1, "before": b, "after": a}
            for i, (b, a) in enumerate(zip_longest(before.splitlines(), after.splitlines(), fillvalue="")) if b != a
        ]
    return record

def collect_good_fix(file_name: str, before: str, after: str, 
                     stage: str = "", error_msg: str = "", model: str = ""):
    record = _build_diff_record(file_name, before, after, model)
    record["stage"] = stage
    record["error_fixed"] = classify_error(error_msg)
    _append_jsonl("good_fixes", record)

def collect_good_change(file_name: str, before: str, after: str,
                        before_speedup: float, after_speedup: float, model: str = ""):
    record = _build_diff_record(file_name, before, after, model)
    record["before_speedup"] = before_speedup
    record["after_speedup"] = after_speedup
    _append_jsonl("good_changes", record)

def log_error(file_name: str, error_msg: str, code: str = "", model: str = ""):
    err_type = classify_error(error_msg)
    key = (err_type, model)
    _seen_errors[key] += 1
    _save_seen_errors()  # Persist after each update
    
    if err_type == 'other' or _seen_errors[key] == 1:
        _append_jsonl("error_samples", {
            "file": file_name,
            "error_type": err_type,
            "error_msg": error_msg[:1000],
            "code": code,
            "model": model
        })

def get_error_stats() -> dict:
    return dict(_seen_errors)
