import torch

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """
    Resize tensor using interpolation.
    
    Design notes:
    - `interpolate` is a core primitive for resizing images / feature maps.
    - We test multiple interpolation modes (nearest, bilinear, bicubic).
    - We test both `size` and `scale_factor` parameterizations.
    - This operator appears frequently in vision pipelines (upsampling, FPN, etc.).
    
    Args:
        input (Tensor): Input tensor of shape (N, C, H, W) or (N, C, D, H, W)
        size (int or tuple): Output spatial size
        scale_factor (float or tuple): Multiplier for spatial size
        mode (str): 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear'
        align_corners (bool): If True, align corner pixels
        
    Returns:
        Tensor: Resized tensor
    """
    return torch.nn.functional.interpolate(input, size=size, scale_factor=scale_factor, 
                                           mode=mode, align_corners=align_corners)

##################################################################################################################################################


import torch

def test_interpolate():
    results = {}

    # Test case 1: Upsample with nearest neighbor
    input1 = torch.randn(2, 3, 4, 4, device='cuda')
    results["test_case_1"] = interpolate(input1, size=(8, 8), mode='nearest')

    # Test case 2: Scale by factor
    input2 = torch.randn(2, 3, 8, 8, device='cuda')
    results["test_case_2"] = interpolate(input2, scale_factor=2.0, mode='bilinear', align_corners=False)

    # Test case 3: Bicubic interpolation
    input3 = torch.randn(2, 3, 8, 8, device='cuda')
    results["test_case_3"] = interpolate(input3, size=(16, 16), mode='bicubic', align_corners=False)

    return results

test_results = test_interpolate()
