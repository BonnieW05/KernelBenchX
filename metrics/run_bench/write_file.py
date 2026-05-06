import os
import json
import argparse
import shutil

golden_metrics_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "golden_metrics")
golden_metrics_list = os.listdir(golden_metrics_folder)

def write_file(input_folder_path, results_path, exe_jsonl=None):
    tab = ' ' * 4
    script_dir = os.environ.get("KERNELBENCHX_SCRIPT_DIR", "./tmp")
    log_dir = os.environ.get("KERNELBENCHX_LOG_DIR", "./logs")
    
    # Absolute path of metrics/ (used for sys.path wiring in generated perf scripts)
    perf_metrics_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    if os.path.exists(script_dir):
        shutil.rmtree(script_dir, ignore_errors=True)
    os.makedirs(script_dir, exist_ok=True)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)

    if os.path.exists(results_path):
        shutil.rmtree(results_path, ignore_errors=True)
    os.makedirs(results_path, exist_ok=True)

    # Load the list of exe-passed operator files (optional filter).
    exe_passed_files = set()
    if exe_jsonl and os.path.exists(exe_jsonl):
        with open(exe_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("ok"):
                    exe_passed_files.add(record["file"])
        print(f"Loaded {len(exe_passed_files)} exe-passed files from {exe_jsonl}")
    
    input_file_list = []
    for root, _, files in os.walk(input_folder_path):
        for f in files:
            if not f.endswith(".py"):
                continue
            # If exe results are provided, only process passed files.
            if exe_jsonl and f not in exe_passed_files:
                continue
            input_file_list.append(os.path.relpath(os.path.join(root, f), input_folder_path))

    for file in input_file_list:
            op = os.path.splitext(os.path.basename(file))[0]
            perf_file_name = op + "_perf.py"
            if perf_file_name not in golden_metrics_list:
                print(f"Skipping {op}: missing template {perf_file_name}")
                continue
            
            # Extract implementation-only code (strip test segment after the separator).
            source_file = os.path.join(input_folder_path, file)
            with open(source_file, 'r', encoding='utf-8') as sf:
                full_code = sf.read()
                # Only take the operator implementation segment (before the separator line).
                impl_code = full_code.split("#" * 146)[0].strip()
            
            # Save the clean implementation to script_dir so perf templates can import it.
            clean_op_file = os.path.join(script_dir, f"{op}.py")
            os.makedirs(os.path.dirname(clean_op_file) or script_dir, exist_ok=True)
            with open(clean_op_file, 'w', encoding='utf-8') as cf:
                cf.write(impl_code)
            
            with open(os.path.join(golden_metrics_folder, perf_file_name), "r") as f:
                # golden_metrics = f.read()
                lines = f.readlines()
                # print(lines)
                updated_lines = []
                for line in lines:
                    # Fix import paths: replace `from kernelbenchx.Category.op import fn`
                    # with a local import from the cleaned op module under script_dir.
                    if "from kernelbenchx." in line and " import " in line:
                        # Preserve original indentation (some templates import inside blocks).
                        import_match = line.strip().split(" import ")
                        if len(import_match) == 2:
                            func_name = import_match[1].strip()
                            indent = line[: len(line) - len(line.lstrip())]
                            line = f"{indent}from {op} import {func_name}\n"
                    elif line == "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n":
                        # Prepend script_dir so the cleaned operator modules are importable.
                        updated_lines.append(f"sys.path.insert(0, '{script_dir}')\n")
                        updated_lines.append(f"sys.path.append('{perf_metrics_dir}')\n")
                        continue
                    
                    # Historical typo: TorchBench_v1 → kernelbenchx.
                    line = line.replace("from TorchBench_v1.", "from kernelbenchx.")
                    line = line.replace("op_perf.get_do_bench_config()", "op_perf.get_do_bench_config(warmup=100, rep=1000)")

                    stripped = line.lstrip()
                    if stripped.startswith("folder_path ="):
                        indent = line[: len(line) - len(stripped)]
                        line = indent + f'folder_path = "{results_path}"\n'

                    updated_lines.append(line)
                golden_metrics = "".join(updated_lines)

            with open(os.path.join(script_dir, perf_file_name), "w") as f:
                f.write(golden_metrics)
            
def parse_args():
    parser = argparse.ArgumentParser(description='write_file')
    parser.add_argument('--input_folder_path', type=str, help='input_folder_path')
    parser.add_argument('--results_path', type=str, help='results_path')
    parser.add_argument('--exe_jsonl', type=str, default=None, help='exe results jsonl to filter passed files')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    input_folder_path = args.input_folder_path
    results_path = args.results_path
    exe_jsonl = args.exe_jsonl
    write_file(input_folder_path, results_path, exe_jsonl)