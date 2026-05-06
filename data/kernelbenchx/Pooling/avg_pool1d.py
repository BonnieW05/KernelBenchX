import torch

def avg_pool1d(input, kernel_size, stride=None, padding=0):
    """
    1D average pooling operation.
    
    Args:
        input (Tensor): Input tensor of shape (N, C, L)
        kernel_size (int): Size of pooling window
        stride (int): Stride of pooling window
        padding (int): Padding to add
        
    Returns:
        Tensor: Pooled output
    """
    return torch.nn.functional.avg_pool1d(input, kernel_size, stride=stride, padding=padding)

##################################################################################################################################################


import torch

def test_avg_pool1d():
    results = {}

    # Test case 1: Basic pooling on sequence
    input1 = torch.randn(2, 4, 16, device='cuda')
    results["test_case_1"] = avg_pool1d(input1, kernel_size=2)

    # Test case 2: With stride
    input2 = torch.randn(2, 4, 32, device='cuda')
    results["test_case_2"] = avg_pool1d(input2, kernel_size=4, stride=2)

    # Test case 3: With padding
    input3 = torch.randn(2, 4, 16, device='cuda')
    results["test_case_3"] = avg_pool1d(input3, kernel_size=3, stride=1, padding=1)

    return results

test_results = test_avg_pool1d()
