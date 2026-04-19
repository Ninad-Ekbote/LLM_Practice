import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token/Word Embedding Layer
    Maps token IDs to learned embedding vectors
    
    This is TRAINABLE - the model learns these embeddings
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, token_ids):
        """
        Args:
            token_ids: Shape (batch_size, seq_len)
        
        Returns:
            embeddings: Shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding Layer
    Adds position information to embeddings
    
    Can be TRAINABLE (learned) or FIXED (sine/cosine)
    Here we implement TRAINABLE version
    """
    
    def __init__(self, max_seq_length, embedding_dim, learnable=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.learnable = learnable
        
        if learnable:
            # Trainable positional embeddings (like in BERT/GPT)
            self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
        else:
            # Fixed sine/cosine positional encoding (like in original Transformer)
            self.register_buffer('pos_embedding', self._get_sinusoidal_encoding(max_seq_length, embedding_dim))
    
    def _get_sinusoidal_encoding(self, max_seq_length, embedding_dim):
        """
        Create fixed sinusoidal positional encoding
        This is NOT trainable
        """
        pos = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(max_seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe.unsqueeze(0)  # Shape: (1, max_seq_length, embedding_dim)
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: Shape (batch_size, seq_len, embedding_dim)
        
        Returns:
            embeddings + position_embeddings: Shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = embeddings.shape
        
        if self.learnable:
            # Get positional embeddings for current sequence length
            pos_ids = torch.arange(seq_len, device=embeddings.device).unsqueeze(0)  # (1, seq_len)
            pos_emb = self.pos_embedding(pos_ids)  # (1, seq_len, embedding_dim)
        else:
            # Use precomputed sinusoidal encoding
            pos_emb = self.pos_embedding[:, :seq_len, :]  # (1, seq_len, embedding_dim)
        
        return embeddings + pos_emb


class EmbeddingLayer(nn.Module):
    """
    Combined Embedding Layer
    Combines Token Embedding + Positional Embedding
    
    TRAINABLE components:
    - Token embeddings: YES
    - Positional embeddings: YES (if learnable=True) or NO (if learnable=False)
    """
    
    def __init__(self, vocab_size, embedding_dim, max_seq_length, pos_learnable=True, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_embedding = PositionalEmbedding(max_seq_length, embedding_dim, learnable=pos_learnable)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
    
    def forward(self, token_ids):
        """
        Args:
            token_ids: Shape (batch_size, seq_len)
        
        Returns:
            embeddings: Shape (batch_size, seq_len, embedding_dim)
        """
        # Get token embeddings
        token_emb = self.token_embedding(token_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Add positional embeddings
        embeddings = self.pos_embedding(token_emb)  # (batch_size, seq_len, embedding_dim)
        
        # Apply dropout
        return self.dropout(embeddings)
