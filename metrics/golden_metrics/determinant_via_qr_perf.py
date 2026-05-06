import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.LinearAlgebra.determinant_via_qr import determinant_via_qr
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('determinant_via_qr', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for exp in range(4, 14):
            n = 2 ** exp
            input_tensor = torch.randn((n, n), dtype=self.dtype)
            self.input_tensors.append(input_tensor)
    
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return determinant_via_qr(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        n = input_tensor.size(0)
        element_size = input_tensor.element_size()
        input_bytes = n * n * element_size
        output_bytes = 1 * element_size
        total_bytes = input_bytes + output_bytes
        return total_bytes / (runtime / 1000) / 1e9  # GB/s
    
    def get_tflops(self, input_tensor, runtime):
        n = input_tensor.size(0)
        flops = 2 * (n ** 3)
        return flops / (runtime / 1000) / 1e12  # TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
