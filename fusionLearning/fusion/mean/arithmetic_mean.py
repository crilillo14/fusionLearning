"""
Given a 4d tensor of shape (N, C, H, W), 

returns the pixel-wise 3D (really 2D because C is 1) arithmetic mean of the tensor.
"""

def arithmetic_mean(tensor : torch.Tensor) -> torch.Tensor:
    return torch.mean(tensor, dim=0)
