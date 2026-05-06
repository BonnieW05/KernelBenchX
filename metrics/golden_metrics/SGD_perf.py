import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from kernelbenchx.Optimizer.SGD import sgd_step
except ModuleNotFoundError:
    import importlib.util

    _module_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'data',
            'kernelbenchx',
            'Optimizer',
            'SGD.py',
        )
    )
    _spec = importlib.util.spec_from_file_location('kernelbenchx_Optimizer_SGD', _module_path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_mod)
    sgd_step = _mod.sgd_step
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('SGD', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        element_size = torch.empty((), dtype=dtype).element_size()
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
        else:
            free_bytes = 2 ** 30

        bytes_per_elem_alloc = 4 * element_size  # param, grad, momentum_buffer, updated param
        max_elements = int((free_bytes * 0.5) // max(bytes_per_elem_alloc, 1))
        for i in range(10, 29):
            size = 2 ** i
            if size > max_elements:
                break
            self.input_tensors.append(size)
        if len(self.input_tensors) == 0:
            self.input_tensors.append(2 ** 10)
    
    def to_cuda(self, input_tuple):
        size = input_tuple
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        param = torch.randn(size, device='cuda', dtype=dtype)
        grad = torch.randn(size, device='cuda', dtype=dtype)
        momentum_buffer = torch.zeros(size, device='cuda', dtype=dtype)
        return (param, grad, momentum_buffer)
        
    def call_op(self, input_tuple):
        param, grad, momentum_buffer = input_tuple
        return sgd_step(param, grad, momentum_buffer)

    def get_gbps(self, input_tuple, runtime):
        param, grad, momentum_buffer = input_tuple
        num_elements = param.numel()
        # Read: param, grad, momentum_buffer; Write: param, momentum_buffer
        total_bytes = num_elements * 5 * param.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        param, grad, momentum_buffer = input_tuple
        num_elements = param.numel()
        # Basic SGD with momentum: ~6 operations per element
        flops = num_elements * 6
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS
    


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=25, rep=100)
    op_perf.run_benchmark()
