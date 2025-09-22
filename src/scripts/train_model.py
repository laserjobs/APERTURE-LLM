# src/scripts/train_model.py
import torch
import torch.nn.functional as F
import yaml
from types import SimpleNamespace
from tqdm import tqdm

import sys
import os

# Add src/aperture_core to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aperture_core.model import APERTURE_LLM
from aperture_core.utils import get_batch, CharTokenizer, set_seed # Import set_seed

def train(config):
    set_seed(config.training.seed) # Set seed at the start of training
    device = 'cuda'  if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load data and tokenizer
    # For prototype: use dummy data (e.g., repeating simple text)
    # In a real scenario, load large datasets of raw text/images/audio
    dummy_text = "This is a simple text string for demonstration. The APERTURE-LLM aims to be the best LLM available. " * 500
    tokenizer = CharTokenizer()
    data = torch.tensor(tokenizer.encode(dummy_text), dtype=torch.long)
    
    # Update vocab_size in config based on actual tokenizer vocab size
    config.model.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {config.model.vocab_size}")
    print(f"Dummy data size: {len(data)} characters")

    # 2. Initialize Model
    model = APERTURE_LLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    # 3. Training Loop
    model.train()
    for epoch in range(config.training.num_epochs):
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        for iter_step in tqdm(range(100)): # Simulate 100 batches per epoch for prototype
            xb, yb = get_batch(data, config.model.block_size, config.training.batch_size, device)

            # Forward pass
            # For prototype, we're only using raw_text_input
            logits = model(xb, focus_strength=0.5) # Use a fixed focus_strength for training

            # Calculate loss
            # Reshape logits to (N*T, V) and targets to (N*T) for CrossEntropyLoss
            B, T, C_vocab = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C_vocab), yb.view(B*T))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_step % config.training.eval_interval == 0:
                print(f"  Loss at iter {iter_step}: {loss.item():.4f}")
        
    print("Training finished.")
    # 4. Save Model
    torch.save(model.state_dict(), f"aperture_llm_model_epoch_{config.training.num_epochs}.pt")
    print(f"Model saved to aperture_llm_model_epoch_{config.training.num_epochs}.pt")

if __name__ == "__main__":
    import argparse

parser = argparse.ArgumentParser(description="Train APERTURE-LLM.")
parser.add_argument('--config', type=str, default='src/config/model_config.yaml',
                    help='Path to the model configuration YAML file.')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config_dict = yaml.safe_load(f)

# Convert dict to SimpleNamespace for easy attribute access
config = SimpleNamespace(**config_dict)
config.model = SimpleNamespace(**config.model)
config.raw_encoder = SimpleNamespace(**config.raw_encoder)
config.raw_encoder.text = SimpleNamespace(**config.raw_encoder.text)
# Ensure image/audio are SimpleNamespace if they exist, otherwise default to None
config.raw_encoder.image = SimpleNamespace(**config.raw_encoder.image) if hasattr(config.raw_encoder, 'image') and config.raw_encoder.image else None
config.raw_encoder.audio = SimpleNamespace(**config.raw_encoder.audio) if hasattr(config.raw_encoder, 'audio') and config.raw_encoder.audio else None

config.dynamic_resolution = SimpleNamespace(**config.dynamic_resolution)
config.output_convergence = SimpleNamespace(**config.output_convergence)
config.training = SimpleNamespace(**config.training)

train(config)
