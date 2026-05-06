import torch

def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    """
    Generates a tensor with random numbers from a uniform distribution on the interval [0, 1).

    Args:
        size (int...): A sequence of integers defining the shape of the output tensor.
        generator (torch.Generator, optional): A pseudorandom number generator for sampling.
        out (Tensor, optional): The output tensor.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
        layout (torch.layout, optional): The desired layout of returned Tensor.
        device (torch.device, optional): The desired device of returned tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor.
        pin_memory (bool, optional): If set, returned tensor would be allocated in the pinned memory (CPU only).

    Returns:
        Tensor: A tensor of shape `size` with random numbers in the interval [0, 1).
    """
    return torch.rand(*size, generator=generator, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory)

##################################################################################################################################################


import torch

def test_rand():
    # Align with EVAL/1_exe_acc.py: same KERNELBENCHX_SEED before any CUDA RNG / Generator use.
    import os
    _seed = int(os.environ.get("KERNELBENCHX_SEED", "0"))
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
        torch.cuda.synchronize()

    results = {}

    # Test case 1: Basic usage with size only and fixed seed
    gen1 = torch.Generator(device='cuda')
    gen1.manual_seed(42)
    results["test_case_1"] = rand(2, 3, generator=gen1, device='cuda')

    # Test case 2: Specifying dtype with fixed seed
    gen2 = torch.Generator(device='cuda')
    gen2.manual_seed(43)
    results["test_case_2"] = rand(2, 3, dtype=torch.float64, generator=gen2, device='cuda')

    # Test case 3: Using a generator with specific seed
    gen3 = torch.Generator(device='cuda')
    gen3.manual_seed(42)
    results["test_case_3"] = rand(2, 3, generator=gen3, device='cuda')

    # Test case 4: Requires gradient with fixed seed
    gen4 = torch.Generator(device='cuda')
    gen4.manual_seed(44)
    results["test_case_4"] = rand(2, 3, requires_grad=True, generator=gen4, device='cuda')

    # Test with multiple seeds - all using generators for determinism
    for seed in (0, 7, 123):
        g = torch.Generator(device='cuda')
        g.manual_seed(seed)
        results[f"test_random_seed_{seed}"] = rand(128, 256, generator=g, device='cuda')

    return results

test_results = test_rand()