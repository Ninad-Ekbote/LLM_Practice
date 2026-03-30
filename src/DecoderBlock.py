import torch
import torch.nn as nn
from attention.multihead_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """A single transformer block with multi-head attention and feed-forward network"""

    def __init__(self, d_in, d_hidden, num_heads, context_length, dropout=0.5, qkv_bias=False):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_in)
        
        # Multi-head attention
        self.mha = MultiHeadAttention(
            d_in=d_in,
            d_out=d_in,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
            qkv_bias=qkv_bias
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_in),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, use_cache=False):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            use_cache: Whether to use KV cache for efficient inference
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_in)
            If use_cache=True, returns (output, cache)
        """
        # Multi-head attention with residual connection (pre-norm)
        attn_output = self.mha(self.norm1(x), use_cache=use_cache)
        
        if use_cache:
            attn_output, cache = attn_output
            x = x + self.dropout(attn_output)
        else:
            x = x + self.dropout(attn_output)
        
        # Feed-forward network with residual connection (pre-norm)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
        
        if use_cache:
            return x, cache
        return x
    
    def clear_cache(self):
        """Clear the KV cache in the attention layer"""
        self.mha.clear_cache()


class DecoderBlock(nn.Module):
    """Decoder with 5 stacked transformer blocks"""
    
    def __init__(self, d_in, d_hidden, num_heads, context_length, dropout=0.5, qkv_bias=False, num_blocks=5):
        super().__init__()
        
        # Create 5 transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_in=d_in,
                d_hidden=d_hidden,
                num_heads=num_heads,
                context_length=context_length,
                dropout=dropout,
                qkv_bias=qkv_bias
            )
            for _ in range(num_blocks)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_in)
    
    def forward(self, x, use_cache=False):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            use_cache: Whether to use KV cache for efficient inference
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_in)
            If use_cache=True, returns (output, caches) where caches is a list of cache tuples
        """
        caches = []
        
        # Pass input through all 5 transformer blocks
        for block in self.blocks:
            if use_cache:
                x, cache = block(x, use_cache=True)
                caches.append(cache)
            else:
                x = block(x, use_cache=False)
        
        # Apply final layer normalization
        x = self.final_norm(x)
        
        if use_cache:
            return x, caches
        return x
    
    def clear_cache(self):
        """Clear the KV cache in all transformer blocks"""
        for block in self.blocks:
            block.clear_cache()
