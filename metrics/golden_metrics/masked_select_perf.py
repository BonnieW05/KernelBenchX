import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Index.masked_select import masked_select
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('masked_select', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        dtype = self.dtype if self.dtype is not None else torch.float32

        # Stream compaction / boolean indexing: keep a fixed 50% mask density to reduce variance.
        for exp in range(12, 23):
            n = 2 ** exp
            x = torch.randn(n, dtype=dtype)
            mask = (torch.arange(n) % 2 == 0)
            self.input_tensors.append((x, mask))

    def to_cuda(self, input_tuple):
        x, mask = input_tuple
        return (x.cuda(), mask.cuda())

    def call_op(self, input_tuple):
        return masked_select(*input_tuple)

    def get_gbps(self, input_tuple, runtime):
        x, mask = input_tuple
        element_size = x.element_size()
        out_elems = int(mask.sum().item())
        bytes_in = x.numel() * element_size
        bytes_mask = mask.numel() * mask.element_size()
        bytes_out = out_elems * element_size
        total_bytes = bytes_in + bytes_mask + bytes_out
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        return 0.0


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
