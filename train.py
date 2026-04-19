"""
Example training script for TinyShakespeare with Transformer
"""
import torch
import torch.nn as nn
from src.data_loader import create_dataloaders
from src.transformer import TransformerBlock
from src.embedding import EmbeddingLayer


def train():
    # Configuration
    DATA_PATH = "data/tinyshakespeare.txt"
    BATCH_SIZE = 32
    CONTEXT_LENGTH = 128
    D_IN = 256
    D_HIDDEN = 1024
    NUM_HEADS = 8
    NUM_BLOCKS = 4
    LEARNING_RATE = 0.001
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Load data
    print(f"Loading TinyShakespeare dataset with BPE tokenization...")
    train_loader, val_loader, vocab_size, chars = create_dataloaders(
        DATA_PATH,
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH
    )
    
    print(f"Vocab size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Build model
    print(f"Building model...")
    embedding = EmbeddingLayer(
        vocab_size=vocab_size,
        embedding_dim=D_IN,
        max_seq_length=CONTEXT_LENGTH,
        pos_learnable=False,  # Fixed sine/cosine positional encoding
        dropout=0.1
    )
    
    model = nn.Sequential(
        embedding,
        *[TransformerBlock(D_IN, D_HIDDEN, NUM_HEADS, CONTEXT_LENGTH) 
          for _ in range(NUM_BLOCKS)],
        nn.Linear(D_IN, vocab_size)
    ).to(DEVICE)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    print("Training complete!")
    torch.save(model.state_dict(), "models/transformer_tinyshakespeare.pt")
    print("Model saved to models/transformer_tinyshakespeare.pt")


if __name__ == "__main__":
    train()
