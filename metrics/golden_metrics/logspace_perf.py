import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernelbenchx.Random.logspace import logspace
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('logspace', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            steps = 2 ** i
            self.input_tensors.append(steps)

    def to_cuda(self, input_steps):
        return input_steps

    def call_op(self, steps):
        return logspace(
            start=0.0, 
            end=10.0, 
            steps=steps, 
            dtype=self.dtype, 
            device="cuda"
        )

    def get_gbps(self, steps, runtime):
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        total_bytes = steps * element_size
        gbps = total_bytes / (runtime / 1000) / 1e9
        return gbps

    def get_tflops(self, steps, runtime):
        flops = steps
        tflops = flops / (runtime / 1000) / 1e12
        return tflops
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
