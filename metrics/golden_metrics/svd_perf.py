import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.LinearAlgebra.svd import svd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('svd', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 9):
            size = 2 ** i
            input_tensor = torch.rand((size, size), dtype=self.dtype)
            self.input_tensors.append(input_tensor)
            
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return svd(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        if len(input_tensor.shape) == 2:
            batch, m, n = 1, *input_tensor.shape
        else:
            batch, m, n = input_tensor.shape
        
        element_size = input_tensor.element_size()
        k = min(m, n)
        
        input_bytes = batch * m * n * element_size
        
        # full_matrices=True (perf default): U:(m,m), S:(k), Vh:(n,n)
        output_bytes = batch * (m * m + k + n * n) * element_size
        
        total_bytes = input_bytes + output_bytes
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tensor, runtime):
        if len(input_tensor.shape) == 2:
            batch, m, n = 1, *input_tensor.shape
        else:
            batch, m, n = input_tensor.shape
        
        k = min(m, n)
        
        if m >= n:
            flops_per_matrix = 4 * m * n**2 - (4/3) * n**3
        else:
            flops_per_matrix = 4 * n * m**2 - (4/3) * m**3
        
        total_flops = batch * flops_per_matrix
        return total_flops / (runtime / 1000) / 1e12



if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
