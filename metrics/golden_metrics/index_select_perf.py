import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Index.index_select import index_select
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('index_select', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        dtype = self.dtype if self.dtype is not None else torch.float32

        # Simulate embedding lookup / row selection: select K rows from [N, D]
        D = 256
        for exp in range(12, 18):
            N = 2 ** exp
            K = max(1, N // 8)
            x = torch.randn(N, D, dtype=dtype)
            idx = torch.randint(0, N, (K,), dtype=torch.long)
            self.input_tensors.append((x, 0, idx))

    def to_cuda(self, input_tuple):
        x, dim, idx = input_tuple
        return (x.cuda(), dim, idx.cuda())

    def call_op(self, input_tuple):
        return index_select(*input_tuple)

    def get_gbps(self, input_tuple, runtime):
        x, dim, idx = input_tuple
        # Rough estimate: read selected rows + write output + read idx
        N, D = x.shape
        K = idx.numel()
        element_size = x.element_size()
        bytes_x = K * D * element_size
        bytes_out = K * D * element_size
        bytes_idx = idx.numel() * idx.element_size()
        total_bytes = bytes_x + bytes_out + bytes_idx
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        return 0.0


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
