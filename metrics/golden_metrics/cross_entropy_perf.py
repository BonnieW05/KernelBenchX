import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, 'data'))

from kernelbenchx.Loss.cross_entropy import cross_entropy
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('cross_entropy', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        # Test a range of (batch_size, num_classes) pairs.
        self.input_tensors = []
        for i in range(8, 16):  # batch_size from 256 to 32768
            batch_size = 2 ** i
            num_classes = 1000  # common ImageNet-scale class count
            input_tensor = torch.randn(batch_size, num_classes, dtype=self.dtype)
            target_tensor = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
            self.input_tensors.append((input_tensor, target_tensor))

    def to_cuda(self, input_tuple):
        return (input_tuple[0].cuda(), input_tuple[1].cuda())
    
    def call_op(self, input_tuple):
        reduction = self.kwargs.get('reduction', 'mean')
        return cross_entropy(input_tuple[0], input_tuple[1], reduction=reduction)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, target_tensor = input_tuple
        reduction = self.kwargs.get('reduction', 'mean')

        input_bytes = input_tensor.numel() * input_tensor.element_size()
        target_bytes = target_tensor.numel() * target_tensor.element_size()

        output_element_size = torch.tensor([], dtype=torch.float32).element_size()

        if reduction == 'none':
            output_bytes = target_tensor.numel() * output_element_size
        else:
            output_bytes = output_element_size

        total_bytes = input_bytes + target_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, _ = input_tuple
        batch_size, num_classes = input_tensor.shape

        exp_cost = self.kwargs.get('exp_cost', 8)  # rough compute-cost estimate (same below)
        log_cost = self.kwargs.get('log_cost', 8)

        flops_per_sample = (num_classes - 1) + num_classes + num_classes * exp_cost + (num_classes - 1) + log_cost + 2
        total_flops = batch_size * flops_per_sample
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
