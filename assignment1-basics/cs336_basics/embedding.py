"""
A embedding layer built by pytorch
"""
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(
            self, 
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), 
            requires_grad=True
        )
        # init embedding matrix
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=1.0, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]