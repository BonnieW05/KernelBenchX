import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Math.asin import asin
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('asin', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # Match the reference tests in data/kernelbenchx/Math/asin.py by using float32.
        # Note: libdevice.asin may fail to compile for float16 inputs in Triton kernels.
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = (torch.rand(size, dtype=torch.float32) * 2 - 1)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return asin(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        FLOPS = input_tensor.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    # Many sizes and large tensors: use a lighter do_bench config to avoid exceeding KERNELBENCHX_BENCH_TIMEOUT.
    op_perf.get_do_bench_config(warmup=25, rep=100)
    op_perf.run_benchmark()
