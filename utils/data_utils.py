import torch

def rand_tensor(
    shape,
    *,
    dtype=torch.float32,
    device="cuda",
    mode="standard",
    outlier_prob=0.001,
    outlier_scale=50.0,
    low=-1.0,
    high=1.0,
):
    if mode == "standard":
        return torch.randn(*shape, device=device, dtype=dtype)
    if mode == "uniform":
        return (high - low) * torch.rand(*shape, device=device, dtype=dtype) + low
    if mode == "outlier":
        x = torch.randn(*shape, device=device, dtype=dtype)
        mask = torch.rand(*shape, device=device) < outlier_prob
        n = int(mask.sum().item())
        if n > 0:
            x = x.clone()
            x[mask] = torch.randn(n, device=device, dtype=dtype) * outlier_scale
        return x

def rand_int(shape, *, low, high, device="cuda", dtype=torch.int64):
    return torch.randint(low, high, shape, device=device, dtype=dtype)

def rand_bool(shape, *, device="cuda"):
    return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
