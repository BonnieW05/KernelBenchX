import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Loss.binary_cross_entropy import binary_cross_entropy
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('binary_cross_entropy', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 24):
            size = 2 ** i
            logits = torch.randn(size, dtype=self.dtype)
            inp = torch.sigmoid(logits)
            tgt = torch.randint(0, 2, (size,), dtype=torch.int64).to(dtype=inp.dtype)
            self.input_tensors.append((inp, tgt))

    def to_cuda(self, input_tuple):
        return (input_tuple[0].cuda(), input_tuple[1].cuda())

    def call_op(self, input_tuple):
        reduction = self.kwargs.get('reduction', 'mean')
        return binary_cross_entropy(input_tuple[0], input_tuple[1], reduction=reduction)

    def get_gbps(self, input_tuple, runtime):
        inp, tgt = input_tuple
        reduction = self.kwargs.get('reduction', 'mean')

        inp_bytes = inp.numel() * inp.element_size()
        tgt_bytes = tgt.numel() * tgt.element_size()

        output_element_size = torch.tensor([], dtype=torch.float32).element_size()

        if reduction == 'none':
            output_bytes = inp.numel() * output_element_size
        else:
            output_bytes = output_element_size

        total_bytes = inp_bytes + tgt_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        inp, _ = input_tuple
        log_cost = self.kwargs.get('log_cost', 8)
        flops_per_element = 2 * log_cost + 6
        flops = inp.numel() * flops_per_element
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
