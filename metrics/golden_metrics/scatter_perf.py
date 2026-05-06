import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Index.scatter import scatter
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('scatter', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(8, 14):
            size = 2 ** i
            input_tensor = torch.zeros(size, 128, dtype=self.dtype)
            index = torch.randint(0, size, (size // 2, 128), dtype=torch.long)
            src = torch.randn(size // 2, 128, dtype=self.dtype)
            self.input_tensors.append((input_tensor, 0, index, src))

    def to_cuda(self, input_tuple):
        return tuple(t.cuda() if isinstance(t, torch.Tensor) else t for t in input_tuple)
    
    def call_op(self, input_tuple):
        return scatter(*input_tuple)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, dim, index, src = input_tuple
        num_elements = input_tensor.numel() + index.numel() + src.numel()
        element_size = input_tensor.element_size()
        total_bytes = num_elements * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        return 0.0


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
