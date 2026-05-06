import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Math.chebyshev_polynomial_t import chebyshev_polynomial_t
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, n=3, **kwargs):
        super().__init__('chebyshev_polynomial_t', dtype=dtype, is_backward=is_backward, **kwargs)
        self.n = n

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()

    def call_op(self, input_tensor):
        return chebyshev_polynomial_t(input_tensor, self.n)

    def get_gbps(self, input_tensor, runtime):
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        if self.n <= 1:
            flops_per_element = 0
        else:
            flops_per_element = 3 * (self.n - 1)
        total_flops = input_tensor.numel() * flops_per_element
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
