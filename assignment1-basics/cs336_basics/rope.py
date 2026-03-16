"""
RoPE implementation by PyTorch
"""
import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        powers = torch.arange(0, d_k, 2).float() / d_k
        inv_freqs = 1.0 / (theta**powers)
        i = torch.arange(max_seq_len)
        angles = torch.outer(i, inv_freqs)
        
        # cache cos and sin values
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        if device is not None:
            self.to(device)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # split x into odd and even indices
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # apply RoPE
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x1 * self.cos[token_positions] - x2 * self.sin[token_positions]
        x_rotated[..., 1::2] = x1 * self.sin[token_positions] + x2 * self.cos[token_positions]

        return x_rotated