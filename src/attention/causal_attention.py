import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        q  = self.W_q(x)
        k  = self.W_k(x)
        v  = self.W_v(x)
        context_lenght = x.shape[0]

        attn_scores  = q @ k.T
        attn_scores = attn_scores/self.d_out**0.5
        mask = torch.triu(torch.ones(context_lenght, context_lenght), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask.bool(), float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        print(attn_weights)
        context_vector = attn_weights @ v

        return context_vector