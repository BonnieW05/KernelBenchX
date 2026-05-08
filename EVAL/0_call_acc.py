import json
import os,argparse
import ast
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import re
try:
    from code_quality import analyze_code_quality
    _QUALITY_AVAILABLE = True
except Exception:
    _QUALITY_AVAILABLE = False

sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from quantization_checker import check_quantization_task
from correction_utils import impl_must_export_kernel


_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.normpath(os.path.join(_EVAL_DIR, ".."))

statis_path = os.path.join(_REPO_DIR, "data", "kernelbenchx_v1.json")
py_folder = os.path.join(_REPO_DIR, "data", "kernelbenchx")
py_interpreter = sys.executable


def _parse_gpus_arg(gpus_str: str):
    s = str(gpus_str).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            return [int(x) for x in re.findall(r"\d+", s)]
        except Exception:
            pass
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    return [int(s)]

def extract_functions_and_imports(code):
    # Parse the code into an AST
    tree = ast.parse(code)
    
    functions = []
    imports = []

    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        # Check if the node is a function definition
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        
        # Check if the node is an import statement
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # For 'import' and 'from ... import' statements, we store the original code text
            imports.append(ast.unparse(node))  # Using `unparse` for getting the code text

    return functions, imports

def clear_code(code: str) -> str:
    code = str(code)
    if "```python" in code:
        code = code.split("```python", 1)[-1]
    if "```" in code:
        code = code.split("```", 1)[0]
    code = code.replace("<|im_end|>", "").replace("<|EOT|>", "")
    return code.strip()

def get_test(folder: str, files: list) -> list[str]:
    test = []
    for f in files:
        # First try full relative path (e.g., Category/file.py).
        path = os.path.join(folder, f)
        
        # If that fails, fall back to searching by basename across the dataset.
        if not os.path.exists(path):
            basename = os.path.basename(f)
            found = False
            for root, dirs, filenames in os.walk(folder):
                if basename in filenames:
                    path = os.path.join(root, basename)
                    found = True
                    break
            if not found:
                raise AssertionError(f"{f} not found (tried full path and basename search)")
        
        code = open(path, "r", encoding="utf-8").read().split("#"*146)[-1]
        assert "def test_" in code, f"No test_ function in {f}"
        test.append(code)
    
    assert len(files) == len(test)
    return test

def get_corresponding_files(instrus: list) -> list:
    infos = json.loads(open(statis_path, 'r', encoding='utf-8').read())
    files = []
    for instru in instrus:
        f = []
        assert "Functional Description: " in instru and "Wrapper Entry Information:" in instru, ""
        func = instru.split("Functional Description: ")[-1].split("Wrapper Entry Information:")[0].replace("\n", "")
        for item in infos:
            if func in item["description"].replace("\n", ""):
                f.append(item["file"])
        assert len(f) == 1, ""
        files.append(f[0])
    assert len(files) == len(instrus)
    return files

def get_codes_from_py(path):
    """Extract code from a single .py file."""
    code = open(path, 'r', encoding='utf-8').read()
    if "#" * 146 in code:
        code = code.split("#" * 146)[0]
    filename = os.path.basename(path)
    return [code], [filename]

def get_codes_for_test(path):
    if path.endswith(".py"):
        codes, files = get_codes_from_py(path)
        tests = get_test(py_folder, files)
        return codes, tests, files
    
    assert path.endswith(".jsonl"), ""
    data = [json.loads(line) for line in open(path, 'r', encoding='utf-8').readlines()]
    assert len(data) > 0, ""
    assert all("predict" in item for item in data), ""
    codes = [clear_code(item["predict"]) for item in data]

    if all("file" in item for item in data):
        files = [item["file"] for item in data]
        print("file")
    else:
        key = None
        for cand in ("instruction", "prompt", "query"):
            if cand in data[0]:
                key = cand
                break
        if key is None:
            key = list(data[0].keys())[0]
        print(key)
        files = get_corresponding_files([item[key] for item in data])
    
    # Only keep basenames (strip category directories).
    files = [os.path.basename(f) for f in files]
    tests = get_test(py_folder, files)
    # Mark empty predictions for later handling
    codes_with_empty_flag = [(c, c.strip() == "") for c in codes]
    return codes_with_empty_flag, tests, files

def run_script_on_gpu(script_content_tuple, test_content, file_name, tmp_dir, gpu_id):
    """
    Runs a given Python script on a specified GPU.
    script_content_tuple: (script_content, is_empty) tuple
    """
    script_content, is_empty = script_content_tuple
    os.makedirs(tmp_dir, exist_ok=True)
    temp_path = os.path.abspath(os.path.join(tmp_dir, file_name))
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    try:
        # If prediction is empty, skip execution and report failure
        if is_empty:
            stderr = "Generation failed: empty prediction from model"
            print(f"=== Skipping {file_name}: {stderr} ===", flush=True)
            return False, file_name, "", stderr, {}
        
        # Check quantization implementation for quantization tasks
        is_quant_valid, quant_msg = check_quantization_task(script_content, file_name)
        if not is_quant_valid:
            stderr = f"Quantization check failed: {quant_msg}"
            print(f"=== Quantization violation in {file_name}: {quant_msg} ===", flush=True)
            return False, file_name, script_content, stderr, {}

        stem = os.path.splitext(file_name)[0]
        impl = script_content.split("#" * 146)[0] if "#" * 146 in script_content else script_content
        ok_e, msg_e = impl_must_export_kernel(impl, stem)
        if not ok_e:
            print(f"=== Kernel entry check failed {file_name}: {msg_e} ===", flush=True)
            return False, file_name, script_content, msg_e, {}

        with open(temp_path, "w") as temp_file:
            temp_file.write(script_content + "\n" + "#" * 146 + "\n" + test_content)

        # Set GPU device for execution
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Run the temporary Python file
        result = subprocess.run(
            [py_interpreter, temp_path], 
            capture_output=True, 
            text=True,
            env=env,
            cwd=_REPO_DIR,
        )

        success = result.returncode == 0  # Determine if execution was successful

        # Output execution results
        print(f"=== Output for {file_name} on GPU {gpu_id} ===", flush=True)
        print(result.stdout, flush=True)

        print(f"=== Errors for {file_name} on GPU {gpu_id} ===", flush=True)
        print(result.stderr, flush=True)

        stderr = result.stderr[:2000] if result.stderr else ""
        quality = analyze_code_quality(temp_path) if (success and _QUALITY_AVAILABLE) else {}
        return success, file_name, script_content, stderr, quality

    finally:
        pass

def _normalize_filename(filename):
    """Convert bare filename to full path using metadata"""
    metadata = json.loads(open(statis_path, 'r', encoding='utf-8').read())
    basename = os.path.basename(filename)
    for item in metadata:
        if os.path.basename(item["file"]) == basename:
            return item["file"]
    return filename

def run_code_parallel(pred, test, files, tmp_dir="temp", gpus=[0, 1, 2, 3, 4, 5, 6, 7], delete=False, result_jsonl=None):
    """
    Runs code in parallel across multiple GPUs, ensuring each GPU runs one script at a time.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    total_scripts = len(pred)
    correct_count = 0
    ok_save_files = []
    results = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        future_to_file = {
            executor.submit(run_script_on_gpu, p, t, f, tmp_dir, gpus[i % len(gpus)]): f
            for i, (p, t, f) in enumerate(zip(pred, test, files))
        }

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                success, fname, code, stderr, quality = future.result()
                # Always write normalized full paths (with category directories).
                normalized_fname = _normalize_filename(fname)
                results.append({"file": normalized_fname, "ok": success, "code": code, "stderr": stderr, "code_quality": quality})
                if success:
                    correct_count += 1
                    ok_save_files.append(fname)
            except Exception as e:
                print(f"Error processing {file_name}: {e}", flush=True)
                normalized_fname = _normalize_filename(file_name)
                results.append({"file": normalized_fname, "ok": False, "code": "", "stderr": str(e), "code_quality": {}})

    if delete:
        ok_set = set(ok_save_files)
        for root, _, filenames in os.walk(tmp_dir):
            for file in filenames:
                rel_path = os.path.relpath(os.path.join(root, file), tmp_dir)
                if rel_path in ok_set:
                    continue
                try:
                    os.remove(os.path.join(root, file))
                    print(f"Deleted {rel_path}")
                except Exception as e:
                    print(f"Error deleting {rel_path}: {e}")
    # Calculate and print the correct execution rate
    correct_rate = (correct_count / total_scripts) * 100
    print(f"\nCorrect execution rate: {correct_rate:.2f}%", flush=True)
    print(ok_save_files)
    
    if result_jsonl:
        os.makedirs(os.path.dirname(result_jsonl) or ".", exist_ok=True)
        with open(result_jsonl, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return results

def call_4folder(folder, tgt_folder, gpus=[0, 1, 2, 3, 4, 5, 6, 7], result_jsonl=None):
    # Recursively collect .py and .jsonl inputs.
    all_files = []
    for root, dirs, files_in_dir in os.walk(folder):
        for f in files_in_dir:
            if f.endswith(".jsonl") or f.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, f), folder)
                all_files.append(rel_path)
    
    # Accumulate all results into a single list.
    all_results = []
    
    for rel_path in all_files:
        generated_path = os.path.join(folder, rel_path)
        filename = os.path.basename(rel_path)
        
        # For a .py file, match its test data by basename.
        if generated_path.endswith(".py"):
            codes, _ = get_codes_from_py(generated_path)
            # Only use basenames; get_test() will search recursively.
            files = [filename]
            tests = get_test(py_folder, files)
            # Convert to the (code, is_empty) tuple format.
            pred = [(c, c.strip() == "") for c in codes]
            test = tests
        else:
            # JSONL file
            pred, test, files = get_codes_for_test(generated_path)
        
        # Output folder name uses the input filename (no nested directory structure).
        target_path = os.path.join(tgt_folder, filename.replace('.py', '').replace('.jsonl', ''))
        
        # Do not write sharded result files; keep a single aggregated results list.
        batch_results = run_code_parallel(pred, test, files, tmp_dir=target_path, delete=True, gpus=gpus, result_jsonl=None)
        all_results.extend(batch_results)
        print(f"Above is call test for {filename}")
        print("===="*20)
    
    # Write a single results_call.jsonl file.
    if result_jsonl:
        os.makedirs(os.path.dirname(result_jsonl) or ".", exist_ok=True)
        with open(result_jsonl, 'w', encoding='utf-8') as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\n✓ Unified call results saved: {result_jsonl} ({len(all_results)} tests)")
    
    return all_results

def call_4file(path, tgt_path, gpus=[0], result_jsonl=None):
    pred, test, files = get_codes_for_test(path)
    # get_codes_for_test() returns a list[str] for .py input, but run_script_on_gpu()
    # expects each pred item to be a (script_content, is_empty) tuple.
    if len(pred) > 0 and isinstance(pred[0], str):
        pred = [(c, str(c).strip() == "") for c in pred]
    run_code_parallel(pred, test, files, tmp_dir=tgt_path, delete=True, gpus=gpus, result_jsonl=result_jsonl)
    basename = os.path.basename(path).replace('.jsonl', '').replace('.py', '')
    print(f"Above is call test for {basename}")
    print("===="*40)

def main():
    parser = argparse.ArgumentParser(description="Call Triton-G operator.")
    parser.add_argument('--source', type=str, required=True, help="Source directory or jsonl file for test.")
    parser.add_argument('--target', type=str, required=True, help="Target directory to save the output.")
    parser.add_argument('--GPUs', type=str, required=True, help="number of GPUs available.")
    parser.add_argument('--result_jsonl', type=str, default=None, help="Path to output structured results.")

    args = parser.parse_args()
    gpus = _parse_gpus_arg(args.GPUs)
    
    if os.path.isdir(args.source):
        call_4folder(args.source, args.target, gpus, args.result_jsonl)
    else:
        assert os.path.isfile(args.source), ""
        call_4file(args.source, args.target, gpus, args.result_jsonl)


if __name__ == "__main__":
    main()
