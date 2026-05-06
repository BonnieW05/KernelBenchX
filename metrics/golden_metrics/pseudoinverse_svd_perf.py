import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.LinearAlgebra.pseudoinverse_svd import pseudoinverse_svd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('pseudoinverse_svd', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 10):
            size = 2 ** i
            input_tensor = torch.rand((size, size), dtype=torch.float32)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return pseudoinverse_svd(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        m, n = input_tensor.shape[-2], input_tensor.shape[-1]
        total_bytes = (m*n + n*m) * input_tensor.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        n = input_tensor.shape[-1]
        flops = 24 * (n ** 3)
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
