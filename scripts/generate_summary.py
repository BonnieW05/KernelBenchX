#!/usr/bin/env python3
"""Generate an Excel-friendly summary CSV from metrics.json."""
import json
import csv
import sys
import argparse

def generate_summary(metrics_json_path, output_csv_path):
    with open(metrics_json_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    rows = []
    for file_path, data in sorted(metrics.items()):
        row = {
            'file': file_path,
            'call_ok': data.get('call_ok', False),
            'exe_ok': data.get('exe_ok', False),
            'perf_ok': data.get('perf_ok', False),
            'dtype': data.get('dtype', ''),
            'runtime_ms': data.get('runtime_ms', 'N/A'),
            'speedup': data.get('speedup', 'N/A'),
            'gbs': data.get('gbs', 'N/A'),
            'tflops': data.get('tflops', 'N/A'),
            'occupancy': data.get('achieved_occupancy', 'N/A'),
            'maintainability': data.get('maintainability', 'N/A'),
        }
        rows.append(row)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file', 'call_ok', 'exe_ok', 'perf_ok', 
            'dtype', 'runtime_ms', 'speedup', 'gbs', 
            'tflops', 'occupancy', 'maintainability'
        ])
        writer.writeheader()
        writer.writerows(rows)
    
    # Summary statistics
    total = len(rows)
    call_pass = sum(1 for r in rows if r['call_ok'])
    exe_pass = sum(1 for r in rows if r['exe_ok'])
    perf_pass = sum(1 for r in rows if r['perf_ok'])
    
    print(f"✓ Summary CSV generated: {output_csv_path}")
    print(f"  Total: {total}")
    print(f"  Call passed: {call_pass} ({call_pass/total*100:.1f}%)")
    print(f"  Execution passed: {exe_pass} ({exe_pass/total*100:.1f}%)")
    print(f"  Performance passed: {perf_pass} ({perf_pass/total*100:.1f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Excel-friendly summary from metrics.json')
    parser.add_argument('metrics_json', help='Path to metrics.json')
    parser.add_argument('output_csv', help='Path to output summary.csv')
    args = parser.parse_args()
    
    generate_summary(args.metrics_json, args.output_csv)
