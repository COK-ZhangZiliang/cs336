"""
Causal Multi-Head Self-Attention implementation by PyTorch
"""
import torch
from torch import nn

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None, theta: float | None = None, max_seq_len: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        
        from cs336_basics.linear import Linear
        from cs336_basics.rope import RoPE
        self.W_Q = Linear(d_model, d_model, device, dtype)
        self.W_K = Linear(d_model, d_model, device, dtype)
        self.W_V = Linear(d_model, d_model, device, dtype)
        self.W_O = Linear(d_model, d_model, device, dtype)
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        mask = (~torch.triu(torch.ones(Q.shape[-2], K.shape[-2]), diagonal=1).bool()).float()
        
        heads = []
        for head in range(self.num_heads):
            Q_head = Q[..., head * self.d_k:(head + 1) * self.d_k]
            K_head = K[..., head * self.d_k:(head + 1) * self.d_v]
            V_head = V[..., head * self.d_v:(head + 1) * self.d_v]
            if self.rope is not None:
                Q_head = self.rope(Q_head, token_positions)
                K_head = self.rope(K_head, token_positions)
            from cs336_basics.attention import ScaledDotProductAttention
            attention = ScaledDotProductAttention()
            attention_output = attention(Q_head, K_head, V_head, mask)
            heads.append(attention_output)
        
        return self.W_O(torch.cat([head for head in heads], dim=-1))