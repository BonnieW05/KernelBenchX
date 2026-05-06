import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.LinearAlgebra.ldl_factor import ldl_factor
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('ldl_factor', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(5, 11):
            n = 2 ** i
            A = torch.rand(n, n, dtype=self.dtype or torch.float32)
            A = A + A.T
            A += n * torch.eye(n, dtype=self.dtype or torch.float32)
            self.input_tensors.append(A)
    
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        result = ldl_factor(input_tensor)
        torch.cuda.synchronize()
        return result
    
    def get_gbps(self, input_tensor, runtime):
        n = input_tensor.size(0)
        element_size = input_tensor.element_size()
        total_bytes = (n*n*element_size) * 2 + (n * 8)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        n = input_tensor.size(0)
        flops = n**3 / 3
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
