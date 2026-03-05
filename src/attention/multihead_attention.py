import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, num_heads, context_length, dropout =0.5 , qkv_bias=False,):
        super().__init__()
        
        self.q_query = nn.Linear(d_in, d_out, qkv_bias)
        self.k_query = nn.Linear(d_in, d_out, qkv_bias)
        self.v_query = nn.Linear(d_in, d_out, qkv_bias)
        self.nums_heads = num_heads
        self.head_dim = d_out // num_heads
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.drop = nn.Dropout(dropout)
        self.proj_layer = nn.Linear(d_out, d_out, False)
        
        # KV Cache
        self.kv_cache = None

    def forward(self, x, use_cache=False):
        b, num_tokens, d_out = x.shape
        
        query =  self.q_query(x)  # (b, num_tokes, d_out)
        keys = self.k_query(x)
        values = self.v_query(x)

        query = query.view(b, num_tokens,  self.nums_heads, self.head_dim)  # (b, nm_tokens, nums_head, head_dim)
        keys = keys.view(b, num_tokens,  self.nums_heads, self.head_dim)
        values =  values.view(b, num_tokens,  self.nums_heads, self.head_dim)

        query = query.transpose(1,2)  # (b, nums_head, num_tokens, head_dim)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        
        # Use KV cache if available
        if use_cache and self.kv_cache is not None:
            cached_keys, cached_values = self.kv_cache
            keys = torch.cat([cached_keys, keys], dim=2)  # (b, nums_head, total_tokens, head_dim)
            values = torch.cat([cached_values, values], dim=2)
            
        # Update cache
        if use_cache:
            self.kv_cache = (keys, values)

        attn_scores = query @ keys.transpose(2,3) # (b, nums_head, num_tokens, num_tokens)
        attn_scores = attn_scores / (keys.shape[-1]**0.5)

        attn_scores = torch.masked_fill(attn_scores, self.mask == 1, -torch.inf)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        attn_weights = self.drop(attn_weights)
        context_vector = (attn_weights @ values).transpose(1,2) # (b, num_tokens, nums_head, head_dim)
        context_vector = context_vector.contiguous().view(b, num_tokens, d_out)

        context_vector = self.proj_layer(context_vector)

        if use_cache:
            return context_vector, self.kv_cache
        return context_vector
    
    def clear_cache(self):
        """Clear the KV cache"""
        self.kv_cache = None












