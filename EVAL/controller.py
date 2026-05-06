#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from data_collector import collect_good_fix, collect_good_change, log_error
    HAS_COLLECTOR = True
except ImportError:
    HAS_COLLECTOR = False

SPEEDUP_IMPROVE_THRESHOLD = 1.5

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_METADATA_PATH = os.path.join(_EVAL_DIR, "../data/kernelbenchx_v1.json")

def _build_basename_to_fullpath():
    mapping = {}
    if not os.path.exists(_METADATA_PATH):
        return mapping
    try:
        with open(_METADATA_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            full = item.get("file", "")
            if full:
                mapping[os.path.basename(full)] = full
    except Exception:
        pass
    return mapping

_BASE_TO_FULL = None

def _normalize_key(file_key: str) -> str:
    """Normalize a bare filename (e.g. 'sqrt.py') to full relative path (e.g. 'Math/sqrt.py')."""
    global _BASE_TO_FULL
    if os.sep in file_key or "/" in file_key:
        return file_key
    if _BASE_TO_FULL is None:
        _BASE_TO_FULL = _build_basename_to_fullpath()
    return _BASE_TO_FULL.get(file_key, file_key)


class IterController:
    def __init__(self, base_dir: str = "runs", enable_collect: bool = True, use_run_id: bool = True):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(base_dir) / self.run_id if use_run_id else Path(base_dir)
        self.enable_collect = enable_collect and HAS_COLLECTOR
        self.iter = 0
        self.state = defaultdict(lambda: {"call": None, "exe": None, "eff": None})

    def _load_jsonl(self, path: str) -> list:
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]

    def _fallback_before_code(self, file_key: str, preferred_stage: str = "") -> str:
        """Return the best available non-empty historical code for this file."""
        stage_order = [preferred_stage] if preferred_stage else []
        for s in ("call", "exe", "eff"):
            if s not in stage_order:
                stage_order.append(s)

        st = self.state[file_key]
        for stage in stage_order:
            item = st.get(stage)
            if item and item.get("code"):
                return item["code"]
        return ""

    def process_call_acc(self, results: list, model: str = ""):
        if not self.enable_collect:
            return
        for r in results:
            f = _normalize_key(r["file"])
            prev = self.state[f]["call"]
            # fail -> success: collect fix
            if prev and not prev.get("ok", False) and r.get("ok", False):
                before_code = prev.get("code") or self._fallback_before_code(f, preferred_stage="call")
                collect_good_fix(f, before_code, r.get("code", ""), stage="call_acc", 
                               error_msg=prev.get("stderr", ""), model=model)
            if not r.get("ok", False):
                log_error(f, r.get("stderr", ""), r.get("code", ""), model=model)
            self.state[f]["call"] = r

    def process_exe_acc(self, results: list, model: str = ""):
        if not self.enable_collect:
            return
        for r in results:
            f = _normalize_key(r["file"])
            prev = self.state[f]["exe"]
            # fail -> success: collect fix
            if prev and not prev.get("ok", False) and r.get("ok", False):
                before_code = prev.get("code") or self._fallback_before_code(f, preferred_stage="exe")
                collect_good_fix(f, before_code, r.get("code", ""), stage="exe_acc", model=model)
            if not r.get("ok", False):
                log_error(f, r.get("stderr", ""), r.get("code", ""), model=model)
            self.state[f]["exe"] = r

    def process_efficiency(self, results: list, model: str = ""):
        if not self.enable_collect:
            return
        for r in results:
            f = _normalize_key(r["file"])
            if not r.get("ok", False):
                log_error(f, r.get("error", "") or r.get("stderr", ""), r.get("code", ""), model=model)
                continue
            speedup = r.get("speedup", 0)
            prev_eff = self.state[f]["eff"]
            if prev_eff and prev_eff.get("speedup", 0) > 0:
                prev_speedup = prev_eff["speedup"]
                improvement = speedup - prev_speedup
                if improvement > SPEEDUP_IMPROVE_THRESHOLD:
                    collect_good_change(f, prev_eff["code"], r["code"],
                                       prev_speedup, speedup, model=model)
            self.state[f]["eff"] = r

    # gather above
    def process_iter(self, call_jsonl=None, exe_jsonl=None, eff_jsonl=None, model=""):
        if call_jsonl and os.path.exists(call_jsonl):
            self.process_call_acc(self._load_jsonl(call_jsonl), model)
        if exe_jsonl and os.path.exists(exe_jsonl):
            self.process_exe_acc(self._load_jsonl(exe_jsonl), model)
        if eff_jsonl and os.path.exists(eff_jsonl):
            self.process_efficiency(self._load_jsonl(eff_jsonl), model)

    def save_state(self):
        state_file = self.base_dir / "controller_state.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with state_file.open('w') as f:
            json.dump({
                "run_id": self.run_id,
                "iter": self.iter,
                "state": dict(self.state)
            }, f, indent=2)

    def load_state(self):
        state_file = self.base_dir / "controller_state.json"
        if state_file.exists():
            with state_file.open('r') as f:
                data = json.load(f)
                self.iter = data.get("iter", 0)
                self.state = defaultdict(lambda: {"call": None, "exe": None, "eff": None})
                self.state.update(data.get("state", {}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, required=True)
    parser.add_argument('--call', type=str, default=None)
    parser.add_argument('--exe', type=str, default=None)
    parser.add_argument('--eff', type=str, default=None)
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--collect', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir:
        ctrl = IterController(base_dir=args.output_dir, enable_collect=bool(args.collect), use_run_id=False)
    else:
        ctrl = IterController(base_dir="runs", enable_collect=bool(args.collect), use_run_id=True)
    ctrl.load_state()
    ctrl.iter = args.iter
    ctrl.process_iter(args.call, args.exe, args.eff, args.model)
    ctrl.save_state()
