import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.LinearAlgebra.cholesky import cholesky
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('cholesky', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        dtype = torch.float32
        
        for exp in range(8, 14):
            n = 2 ** exp
            L = torch.randn(n, n, dtype=dtype)
            A = L @ L.T + torch.eye(n, dtype=dtype) * 1e-6
            self.input_tensors.append(A)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return cholesky(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        total_bytes = 2 * input_tensor.numel() * input_tensor.element_size()
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        *batch_dims, n, _ = input_tensor.shape
        batch_size = torch.tensor(batch_dims).prod().item() if batch_dims else 1
        flops = batch_size * (n ** 3) / 3
        return flops / (runtime / 1000) / 1e12



if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
