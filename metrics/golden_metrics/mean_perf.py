import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Reduce.mean import mean
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, dim=0, **kwargs):
        super().__init__('mean', dtype=dtype, is_backward=is_backward, **kwargs)
        self.dim = dim

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype or torch.float32)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()

    def call_op(self, input_tensor):
        return mean(input_tensor, dim=self.dim)

    def get_gbps(self, input_tensor, runtime):
        input_numel = input_tensor.numel()
        output_numel = 1
        total_bytes = (input_numel + output_numel) * input_tensor.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        FLOPS = input_tensor.numel() * 2
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
