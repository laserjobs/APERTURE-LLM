# src/scripts/train_model.py
import torch
import torch.nn as nn # Added for nn.CrossEntropyLoss
import torch.optim as optim # Added for optim.AdamW
import yaml
from types import SimpleNamespace
from tqdm import tqdm

import sys
import os

# Add src/aperture_core to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aperture_core.model import APERTURE_LLM
from aperture_core.utils import get_batch, CharTokenizer, set_seed

def train(config):
    set_seed(config.training.seed) # Set seed at the start of training
    device = 'cuda'  if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load data and tokenizer
    tokenizer = CharTokenizer()
    
    # FIX: Updated dummy text to be compatible with 256-char tokenizer
    dummy_text = "This is a simple text string for demonstration. The APERTURE LLM aims to be the best LLM available. It processes raw digital inputs directly. Hello World 123!@#$%^&*()_+-=[]{}|;':\",./<>?~`" * 50
    data = torch.tensor(tokenizer.encode(dummy_text), dtype=torch.long)
    
    # Update vocab_size in config based on actual tokenizer vocab size
    config.model.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {config.model.vocab_size}")
    print(f"Dummy data size: {len(data)} characters")

    # 2. Initialize Model, Optimizer, and Loss Function (added loss function init)
    model = APERTURE_LLM(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss() # Initialize CrossEntropyLoss
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Optional: Debug Tensor Shapes - START
    xb_debug, yb_debug = get_batch(data, config.model.block_size, config.training.batch_size, device)
    print(f"DEBUG: Input batch shape: {xb_debug.shape}, Target batch shape: {yb_debug.shape}") # Expected: (batch_size, block_size) e.g., (8, 256)
    with torch.no_grad():
        logits_debug = model(xb_debug, focus_strength=0.5) 
    print(f"DEBUG: Logits shape: {logits_debug.shape}") # Expected: (batch_size, block_size, vocab_size) e.g., (8, 256, 256)
    # Optional: Debug Tensor Shapes - END

    # 3. Training Loop
    model.train()
    # FIX: Make the number of iterations dynamic
    # Roughly 5x through the data per epoch (adjust multiplier as needed)
    num_iterations = (len(data) - config.model.block_size) // config.training.batch_size 
    if num_iterations == 0: # Ensure at least some iterations for very small dummy text
        num_iterations = 100 
    
    for epoch in range(config.training.num_epochs):
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        for iter_step in tqdm(range(num_iterations), desc=f"Epoch {epoch+1}"): # Use dynamic num_iterations
            xb, yb = get_batch(data, config.model.block_size, config.training.batch_size, device)

            # Forward pass
            logits = model(xb, focus_strength=0.5) # Use a fixed focus_strength for training

            # Calculate loss (using the initialized criterion)
            # Reshape logits to (N*T, V) and targets to (N*T) for CrossEntropyLoss
            B, T, C_vocab = logits.shape
            loss = criterion(logits.view(B*T, C_vocab), yb.view(B*T)) # Use criterion

            # Backward pass and optimize
            # FIX: Use set_to_none=True for memory efficiency
            optimizer.zero_grad(set_to_none=True) 
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

    # FIX: Add error handling for config loading
    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)
    except yaml.YAMLError as e: # Catch specific YAML error
        print(f"Error: Invalid YAML format in {args.config}. Details: {e}")
        sys.exit(1)

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
