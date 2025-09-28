import torch
import numpy as np
import random
import os


# Simple char-level tokenizer for demonstration
class CharTokenizer:
    def __init__(self):
        # Full ASCII (0-255) for vocab_size 256
        self.chars = [chr(i) for i in range(256)]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        # Default to space (ASCII 32) if character is not in vocab.
        # Fallback to index 0 if space itself isn't somehow in vocab (though it should be for 0-255).
        default_char_idx = self.stoi.get(' ', 0)
        return [self.stoi.get(c, default_char_idx) for c in s]

    def decode(self, indices):  # Renamed 'l' to 'indices'
        return "".join([self.itos[i] for i in indices])


# Simple dummy data generator for training
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def set_seed(seed):
    """Set random seed for reproducibility across torch, numpy, and python random."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For deterministic CuDNN operations, which can sometimes impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
