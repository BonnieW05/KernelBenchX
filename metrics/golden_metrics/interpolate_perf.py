import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.SpatialOps.interpolate import interpolate
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('interpolate', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(5, 9):
            size = 2 ** i
            input_tensor = torch.randn(2, 3, size, size, dtype=self.dtype)
            target_size = (size * 2, size * 2)
            self.input_tensors.append((input_tensor, target_size))

    def to_cuda(self, input_tuple):
        return (input_tuple[0].cuda(), input_tuple[1])
    
    def call_op(self, input_tuple):
        return interpolate(input_tuple[0], size=input_tuple[1], mode='bilinear', align_corners=False)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, target_size = input_tuple
        input_elements = input_tensor.numel()
        output_elements = input_tensor.shape[0] * input_tensor.shape[1] * target_size[0] * target_size[1]
        total_elements = input_elements + output_elements
        element_size = input_tensor.element_size()
        total_bytes = total_elements * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, target_size = input_tuple
        output_elements = input_tensor.shape[0] * input_tensor.shape[1] * target_size[0] * target_size[1]
        flops = output_elements * 4
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
