"""Quantization implementation heuristic checker for generated kernels.

Uses lightweight pattern matching to detect evidence of manual quantization logic.
This is a weak filter to exclude trivial non-quantized implementations, 
not a full semantic verifier.
"""

import ast
import re
from typing import Dict, List, Tuple


class QuantizationChecker:
    """Heuristic checker for quantization evidence (not semantic verification)."""
    
    # Forbidden: Direct high-level quantization APIs (hard constraint)
    FORBIDDEN_PATTERNS = [
        r'torch\.quantize_per_tensor',
        r'torch\.quantize_per_channel',
        r'torch\.dequantize',
        r'torch\.int_repr',
        r'torch\.q_scale',
        r'torch\.q_zero_point',
        r'torch\.ops\.quantized',
        r'F\.quantize',
    ]
    
    # Evidence signals for quantization logic (soft scoring)
    EVIDENCE_PATTERNS = {
        # A. Scale derivation / range analysis
        'scale_related': [
            r'(?:max|amax|torch\.(?:max|amax))\s*\(.*abs',  # max(abs(...)) or amax
            r'\.abs\(\)\.[ma]?(?:max|amax)',                  # .abs().max() or .abs().amax()
            r'(?:qmax|q_max|quantize_max)',                    # qmax variable
            r'/\s*(?:127|255|7|15|31)(?:\.0)?(?!\d)',        # division by quant range
            r'\*\s*\(1\.0?\s*/\s*(?:127|255|7)\)',          # inv_scale pattern
            r'(?:scale|inv_scale|zero_point)\s*=',            # explicit scale variable
            r'(?:1\s*<<\s*\(?\w+\s*-\s*1\)?\s*-\s*1)',       # (1 << (bits-1)) - 1
            r'\.amax\s*\(\s*dim\s*=',                         # per-channel: amax(dim=...)
        ],
        
        # B. Discretization / quantization operation
        'discretization': [
            r'\.round\(',                                       # .round()
            r'torch\.round\(',                                 # torch.round()
            r'tl\.(?:math\.)?round',                           # triton round
            r'floor\s*\([^)]*\+[^)]*0\.5',                    # floor(x + 0.5) pattern
            r'\.to\s*\(\s*(?:torch|tl)\.(?:u?int8|int16|int32|int4)\s*\)',  # cast to int
            r'\.to\s*\(\s*dtype\s*=\s*torch\.(?:u?int8|int32)\s*\)',
            r'\.int\(\)|.char\(\)|.byte\(\)',                # PyTorch shorthand casts
            r'tl\.cast\s*\([^,]+,\s*tl\.(?:int|uint)\d+',    # Triton cast
            r'\.clamp\s*\([^)]*-?\d+\s*,\s*\d+',             # clamp with numeric range
            r'(?:min|max)\s*\([^)]*,\s*-?\d+\)',             # min/max clipping
        ],
        
        # C. Integer type usage (context signal)
        'integer_context': [
            r'tl\.int(?:8|16|32)',                             # Triton int types
            r'dtype\s*=\s*torch\.int(?:8|16|32)',            # PyTorch int types
            r'(?:accumulator|acc|result).*int32',              # int32 accumulator mention
            r'tl\.dot\s*\([^)]*int',                          # Triton dot with int
        ],
        
        # D. Dequantization / rescaling
        'rescaling': [
            r'\*\s*(?:scale|inv_scale|out_scale|scale_[a-z])',  # multiply by scale-like var
            r'/\s*(?:inv_scale|scale)',                          # divide by scale
            r'\.to\s*\(\s*torch\.float(?:16|32|64)?\s*\)',    # cast back to float
            r'\.float\(\)',                                      # .float() shorthand
            r'scale_[a-z]\s*\*\s*scale_[a-z]',                  # multi-scale product
        ],
    }
    
    def check(self, code: str, task_name: str) -> Dict:
        """Check for quantization evidence using heuristic scoring.
        
        Args:
            code: Generated kernel code
            task_name: Task name (e.g., 'matmul_w8a8')
            
        Returns:
            {
                'is_valid': bool,
                'score': int,
                'violations': List[str],
                'evidence': Dict[str, int],
                'matched_patterns': List[str],
                'message': str
            }
        """
        violations = []
        evidence_counts = {cat: 0 for cat in self.EVIDENCE_PATTERNS}
        matched_patterns = []
        
        # Hard constraint: Check for forbidden patterns (immediate fail)
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Uses forbidden API: {pattern}")
        
        # If forbidden APIs found, fail immediately
        if violations:
            return {
                'is_valid': False,
                'score': 0,
                'violations': violations,
                'evidence': evidence_counts,
                'matched_patterns': [],
                'message': f"Forbidden API usage: {'; '.join(violations)}",
            }
        
        # Soft scoring: Collect evidence from each category
        for category, patterns in self.EVIDENCE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, code, re.IGNORECASE)
                if matches:
                    evidence_counts[category] += len(matches)
                    matched_patterns.append(f"{category}: {pattern[:50]}...")
        
        # Scoring logic: weighted sum
        score = (
            evidence_counts['scale_related'] * 2 +      # Scale is important
            evidence_counts['discretization'] * 3 +     # Rounding/casting is KEY
            evidence_counts['integer_context'] * 1 +    # Supporting evidence
            evidence_counts['rescaling'] * 2            # Dequant matters
        )
        
        # Validation rules (more flexible than before):
        # - Must have some scale-related logic (at least 1)
        # - Must have discretization evidence (at least 2, to avoid false positives)
        # - Recommended to have rescaling (but not strictly required for all tasks)
        
        has_scale = evidence_counts['scale_related'] >= 1
        has_discretization = evidence_counts['discretization'] >= 2
        has_rescaling = evidence_counts['rescaling'] >= 1
        
        # Flexible pass condition:
        is_valid = (
            len(violations) == 0 and
            has_scale and
            has_discretization
            # Note: rescaling not strictly required (e.g., weight-only quant)
        )
        
        # Build message
        if is_valid:
            if has_rescaling:
                confidence = "high"
            else:
                confidence = "moderate (no clear dequant detected)"
            message = f"Quantization evidence detected (score={score}, confidence={confidence})"
        else:
            parts = []
            if not has_scale:
                parts.append("missing scale derivation")
            if not has_discretization:
                parts.append("missing discretization (round/cast)")
            message = f"Insufficient quantization evidence: {', '.join(parts)} (score={score})"
        
        return {
            'is_valid': is_valid,
            'score': score,
            'violations': violations,
            'evidence': evidence_counts,
            'matched_patterns': matched_patterns[:10],  # Limit output
            'message': message,
        }


def check_quantization_task(code: str, task_file: str) -> Tuple[bool, str]:
    """Quick check for quantization tasks.
    
    Args:
        code: Generated kernel code
        task_file: Task file path (e.g., 'Quantization/matmul_w8a8.py')
        
    Returns:
        (is_valid, message)
    """
    if 'Quantization/' not in task_file:
        return True, "Not a quantization task"
    
    task_name = task_file.split('/')[-1].replace('.py', '')
    checker = QuantizationChecker()
    result = checker.check(code, task_name)
    
    return result['is_valid'], result['message']


if __name__ == '__main__':
    # Test case 1: Valid quantization with rounding
    valid_code_1 = """
import triton
import triton.language as tl

@triton.jit
def matmul_w8a8_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
    # Compute scales
    scale_a = tl.max(tl.abs(a)) / 127.0
    scale_b = tl.amax(tl.abs(b), dim=1) / 127  # per-channel
    
    # Quantize to int8 with rounding
    a_int8 = ((a / scale_a).round()).to(tl.int8)
    b_int8 = (b / scale_b).round().clamp(-127, 127).to(tl.int8)
    
    # Int32 accumulation
    c_int32 = tl.dot(a_int8, b_int8)
    
    # Dequantize
    c = c_int32.float() * scale_a * scale_b
"""
    
    # Test case 2: Alternative valid style
    valid_code_2 = """
import torch

def linear_w4a16(input, weight, bias):
    qmax = 7.0
    scale = weight.abs().amax(dim=-1, keepdim=True) / qmax
    w_quant = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
    w_dequant = w_quant.float() * scale
    return torch.matmul(input, w_dequant.T) + bias
"""
    
    # Test case 3: Marginal case (no rounding)
    marginal_code = """
def matmul(a, b):
    scale = a.abs().max() / 127
    a_int = (a / scale).to(torch.int8)  # Missing round!
    return torch.matmul(a_int.float(), b) * scale
"""
    
    # Test case 4: Forbidden API
    invalid_code = """
import torch

def matmul_w8a8(a, b):
    a_q = torch.quantize_per_tensor(a, scale=0.1, zero_point=0, dtype=torch.qint8)
    b_q = torch.quantize_per_tensor(b, scale=0.1, zero_point=0, dtype=torch.qint8)
    return torch.ops.quantized.matmul(a_q, b_q)
"""
    
    # Test case 5: Fake quantization (dead code)
    fake_code = """
def matmul(a, b):
    # Fake patterns to fool checker
    dummy_scale = a.abs().max() / 127
    _ = (a / dummy_scale).round().to(torch.int8)
    # But actually use fp32
    return torch.matmul(a.float(), b.float())
"""
    
    checker = QuantizationChecker()
    
    test_cases = [
        ("Valid (with rounding)", valid_code_1),
        ("Valid (alternative style)", valid_code_2),
        ("Marginal (no rounding)", marginal_code),
        ("Invalid (forbidden API)", invalid_code),
        ("Fake quantization", fake_code),
    ]
    
    for name, code in test_cases:
        print(f"\n=== {name} ===")
        result = checker.check(code, "test_task")
        print(f"Valid: {result['is_valid']}")
        print(f"Score: {result['score']}")
        print(f"Message: {result['message']}")
        print(f"Evidence: {result['evidence']}")
        if result['violations']:
            print(f"Violations: {result['violations']}")
        if result['matched_patterns']:
            print(f"Matched (first 3): {result['matched_patterns'][:3]}")
