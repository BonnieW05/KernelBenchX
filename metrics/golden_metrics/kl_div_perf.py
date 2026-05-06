import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Loss.kl_div import kl_div
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('kl_div', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(8, 14):
            batch = 2 ** i
            classes = 1024
            x = torch.randn(batch, classes, dtype=self.dtype)
            inp = torch.log_softmax(x, dim=-1)
            tgt = torch.softmax(torch.randn(batch, classes, dtype=self.dtype), dim=-1)
            self.input_tensors.append((inp, tgt))

    def to_cuda(self, input_tuple):
        return (input_tuple[0].cuda(), input_tuple[1].cuda())

    def call_op(self, input_tuple):
        return kl_div(input_tuple[0], input_tuple[1])

    def get_gbps(self, input_tuple, runtime):
        inp, tgt = input_tuple
        total_bytes = (inp.numel() + tgt.numel()) * inp.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        inp, tgt = input_tuple
        flops = inp.numel() * 6
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
