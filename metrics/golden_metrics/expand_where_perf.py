import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Index.expand_where import expand_where
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('expand_where', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        dtype = self.dtype if self.dtype is not None else torch.float32
        batch = 256
        for i in range(10, 15):
            size = 2 ** i
            x = torch.randn(1, size, dtype=dtype)
            target_sizes = (batch, size)
            cond = torch.randint(0, 2, (batch, 1), dtype=torch.bool)
            other = torch.randn(batch, size, dtype=dtype)
            self.input_tensors.append((x, target_sizes, cond, other))

    def to_cuda(self, input_tuple):
        x, target_sizes, cond, other = input_tuple
        return (x.cuda(), target_sizes, cond.cuda(), other.cuda())

    def call_op(self, input_tuple):
        x, target_sizes, cond, other = input_tuple
        return expand_where(x, target_sizes, cond, other)

    def get_gbps(self, input_tuple, runtime):
        x, target_sizes, cond, other = input_tuple
        element_size = other.element_size()
        out_numel = other.numel()

        # expand is a view, but where will read expanded values per output element.
        # Approximate bytes: read x(expanded), read other, read cond, write out.
        total_bytes = (3 * out_numel * element_size) + (out_numel * 1)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        x, target_sizes, cond, other = input_tuple
        out_numel = other.numel()
        flops = out_numel
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
