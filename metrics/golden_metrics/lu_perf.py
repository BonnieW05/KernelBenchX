import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.LinearAlgebra.lu import lu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('lu', dtype=dtype, is_backward=is_backward, **kwargs)
        if dtype is None:
            self.dtype = torch.float32

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 13):
            size = 128 * i
            input_tensor = torch.rand(size, size, dtype=self.dtype)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()

    def call_op(self, input_tensor):
        return lu(input_tensor, pivot=True)

    def get_gbps(self, input_tensor, runtime):
        total_bytes = 4 * input_tensor.numel() * input_tensor.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        n = input_tensor.size(0)
        flops = (2.0 / 3) * (n ** 3)
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
