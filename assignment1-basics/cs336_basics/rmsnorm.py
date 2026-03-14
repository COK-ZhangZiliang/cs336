"""
RMSNorm Implementation by PyTorch
"""
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # prevent overflow
        rms = torch.sqrt(torch.mean(x**2, dim = -1, keepdim=True) + self.eps)
        x_normed = x / rms * self.g
        return x_normed.to(in_dtype)