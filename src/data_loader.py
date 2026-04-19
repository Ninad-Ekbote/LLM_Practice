import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class TinyShakespeareDataset(Dataset):
    """Dataset for TinyShakespeare character-level language modeling with BPE tokenization"""
    
    def __init__(self, data_path, tokenizer_path="data/bpe_tokenizer.json", context_length=128, train=True, train_split=0.9):
        self.context_length = context_length
        
        # Download data if not present
        if not os.path.exists(data_path):
            self._download_data(data_path)
        
        # Read the text
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Train or load tokenizer
        self.tokenizer = self._get_tokenizer(tokenizer_path, data_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # Encode the entire text
        encoded = self.tokenizer.encode(self.text)
        self.encoded_text = torch.tensor(encoded.ids, dtype=torch.long)
        
        # Split into train/val
        split_idx = int(len(self.encoded_text) * train_split)
        if train:
            self.data = self.encoded_text[:split_idx]
        else:
            self.data = self.encoded_text[split_idx:]
    
    def _get_tokenizer(self, tokenizer_path, data_path):
        """Get or train BPE tokenizer"""
        # Load if exists
        if os.path.exists(tokenizer_path):
            return Tokenizer.from_file(tokenizer_path)
        
        # Train new tokenizer
        print(f"Training BPE tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=5000,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        )
        
        tokenizer.train([data_path], trainer)
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
        return tokenizer
    
    def __len__(self):
        return len(self.data) - self.context_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1:idx + self.context_length + 1]
        return x, y
    
    def _download_data(self, data_path):
        """Download TinyShakespeare dataset"""
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading TinyShakespeare dataset from {url}...")
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        urllib.request.urlretrieve(url, data_path)
        print(f"Dataset saved to {data_path}")
    
    def decode(self, indices):
        """Convert token indices back to text"""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return self.tokenizer.decode(indices, skip_special_tokens=True)
    
    def encode(self, text):
        """Convert text to token indices"""
        encoded = self.tokenizer.encode(text)
        return torch.tensor(encoded.ids, dtype=torch.long)


def create_dataloaders(data_path, tokenizer_path="data/bpe_tokenizer.json", batch_size=32, context_length=128, num_workers=0):
    """Create train and validation dataloaders for TinyShakespeare with BPE tokenization"""
    
    train_dataset = TinyShakespeareDataset(
        data_path,
        tokenizer_path=tokenizer_path,
        context_length=context_length,
        train=True
    )
    
    val_dataset = TinyShakespeareDataset(
        data_path,
        tokenizer_path=tokenizer_path,
        context_length=context_length,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, train_dataset.vocab_size, train_dataset.chars
