"""
Scaled Dot-Product Attention implementation by PyTorch
"""
import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        d_k = torch.tensor(Q.shape[-1], dtype=torch.float32)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(~mask.bool(), -torch.inf)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, V)