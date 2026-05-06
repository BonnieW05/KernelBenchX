import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Fusion.attention import attention
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('attention', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        batch = 2
        num_heads = 8
        d_k = 64
        for i in range(6, 11):  # seq_len from 64 to 2048
            seq_len = 2 ** i
            q = torch.randn(batch, num_heads, seq_len, d_k, dtype=self.dtype)
            k = torch.randn(batch, num_heads, seq_len, d_k, dtype=self.dtype)
            v = torch.randn(batch, num_heads, seq_len, d_k, dtype=self.dtype)
            self.input_tensors.append((q, k, v))

    def to_cuda(self, input_tuple):
        return tuple(t.cuda() for t in input_tuple)
    
    def call_op(self, input_tuple):
        return attention(*input_tuple, causal=True)
    
    def get_gbps(self, input_tuple, runtime):
        q, k, v = input_tuple
        # Read Q, K, V + Write output
        total_elements = q.numel() * 3 + q.numel()
        total_bytes = total_elements * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        q, k, v = input_tuple
        batch, num_heads, seq_len, d_k = q.shape
        # QK^T: 2 * B*H*S*S*D FLOPs (matmul)
        # Attn @ V: 2 * B*H*S*S*D FLOPs (matmul)
        flops = 4 * batch * num_heads * seq_len * seq_len * d_k
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
