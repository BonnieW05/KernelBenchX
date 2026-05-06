import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Fusion.fused_repeat_interleave_log_softmax import fused_repeat_interleave_log_softmax
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_repeat_interleave_log_softmax', dtype=dtype, is_backward=is_backward, **kwargs)
        self.repeats = 2
        self.dim = 0

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 24):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=torch.float32)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return fused_repeat_interleave_log_softmax(input_tensor, 
                                                 repeats=self.repeats, 
                                                 dim=self.dim)
    
    def get_gbps(self, input_tensor, runtime):
        element_size = input_tensor.element_size()
        input_numel = input_tensor.numel()
        total_bytes = input_numel * element_size * (1 + 3 * self.repeats)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        flops = 3 * input_tensor.numel() * self.repeats  # 3 FLOPs per element
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
