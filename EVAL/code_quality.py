import re
from radon.complexity import cc_visit
from radon.metrics import mi_visit

def analyze_code_quality(file_path):
    """Minimal code quality analysis: complexity, lines, comments."""
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Cyclomatic Complexity
    cc_results = cc_visit(code)
    avg_complexity = sum(r.complexity for r in cc_results) / len(cc_results) if cc_results else 0
    max_complexity = max((r.complexity for r in cc_results), default=0)
    
    # Maintainability Index
    mi_score = mi_visit(code, multi=True)
    
    # Lines and Comments
    lines = code.split('\n')
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    comment_lines = len([l for l in lines if l.strip().startswith('#')])
    comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
    
    return {
        "avg_complexity": round(avg_complexity, 2),
        "max_complexity": max_complexity,
        "maintainability_index": round(mi_score, 2),
        "total_lines": total_lines,
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "comment_ratio": round(comment_ratio, 3)
    }
