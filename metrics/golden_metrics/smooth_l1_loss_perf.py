import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Loss.smooth_l1_loss import smooth_l1_loss
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('smooth_l1_loss', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 25):
            size = 2 ** i
            inp = torch.randn(size, dtype=self.dtype)
            tgt = torch.randn(size, dtype=self.dtype)
            self.input_tensors.append((inp, tgt))

    def to_cuda(self, input_tuple):
        return (input_tuple[0].cuda(), input_tuple[1].cuda())

    def call_op(self, input_tuple):
        return smooth_l1_loss(input_tuple[0], input_tuple[1])

    def get_gbps(self, input_tuple, runtime):
        inp, tgt = input_tuple
        total_bytes = (inp.numel() + tgt.numel()) * inp.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        inp, tgt = input_tuple
        # piecewise (abs, compare, mul/add), still light compute
        flops = inp.numel() * 6
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
