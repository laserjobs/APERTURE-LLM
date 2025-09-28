import argparse
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from aperture_core.model import APERTURE_LLM
from aperture_core.utils import get_batch, CharTokenizer, set_seed

# Add src/aperture_core to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def train(config):
    set_seed(config.training.seed)  # Set seed at the start of training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load data and tokenizer
    tokenizer = CharTokenizer()

    # Further REDUCED multiplier for faster CI execution and to avoid timeouts
    dummy_text = ("This is a simple text string for demonstration. The APERTURE LLM aims to be "
                  "the best LLM available. It processes raw digital inputs directly. "
                  "Hello World 123!@#$%^&*()_+-=[]{}|;':\",./<>?~`") * 10  # Or * 20 if still too fast
    data = torch.tensor(tokenizer.encode(dummy_text), dtype=torch.long)

    # Update vocab_size in config based on actual tokenizer vocab size
    config.model.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {config.model.vocab_size}")
    print(f"Dummy data size: {len(data)} characters")

    # 2. Initialize Model, Optimizer, and Loss Function
    model = APERTURE_LLM(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()
    print(f"APERTURE-LLM Model initialized with "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Optional: Debug Tensor Shapes - START
    # For text-only training, we only provide `xb` to the model.
    xb_debug, yb_debug = get_batch(data, config.model.block_size, config.training.batch_size, device)
    print(f"DEBUG: Input batch shape: {xb_debug.shape}, Target batch shape: {yb_debug.shape}")
    with torch.no_grad():
        logits_debug = model(xb_debug, focus_strength=0.5)  # Text-only
    print(f"DEBUG: Logits shape: {logits_debug.shape}")
    # Optional: Debug Tensor Shapes - END

    # 3. Training Loop
    model.train()
    # Make the number of iterations dynamic based on the larger dummy text
    num_iterations_per_epoch = (len(data) - config.model.block_size) // config.training.batch_size
    if num_iterations_per_epoch == 0:  # Fallback for extremely small datasets
        num_iterations_per_epoch = 100

    for epoch in range(config.training.num_epochs):
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        for iter_step in tqdm(range(num_iterations_per_epoch), desc=f"Epoch {epoch+1}"):
            # For text-only training, we only get `xb` and `yb` from `get_batch`.
            # For multi-modal, you would need to load/generate image/audio data here.
            xb, yb = get_batch(data, config.model.block_size, config.training.batch_size, device)

            # Forward pass (text-only for current prototype)
            logits = model(xb, focus_strength=0.5)

            # Calculate loss
            B, T, C_vocab = logits.shape
            loss = criterion(logits.view(B*T, C_vocab), yb.view(B*T))

            # Backward pass and optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iter_step % config.training.eval_interval == 0:
                print(f"  Loss at iter {iter_step}: {loss.item():.4f}")

    print("Training finished.")
    # 4. Save Model
    model_filename = f"aperture_llm_model_epoch_{config.training.num_epochs}.pt"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train APERTURE-LLM.")
    parser.add_argument('--config', type=str, default='src/config/model_config.yaml',
                        help='Path to the model configuration YAML file.')
    args = parser.parse_args()

    # Error handling for config loading
    config_dict = {}
    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {args.config}. Details: {e}")
        sys.exit(1)

    # Convert dict to SimpleNamespace for easy attribute access
    config = SimpleNamespace(**config_dict)
    config.model = SimpleNamespace(**config.model)
    config.raw_encoder = SimpleNamespace(**config.raw_encoder)
    config.raw_encoder.text = SimpleNamespace(**config.raw_encoder.text)
    # Ensure image/audio are SimpleNamespace if they exist and are enabled in config.
    # Note: yaml.safe_load might omit commented-out sections entirely, so hasattr is key.
    config.raw_encoder.image = (SimpleNamespace(**config.raw_encoder.image)
                                if hasattr(config.raw_encoder, 'image') and config.raw_encoder.image
                                else None)
    config.raw_encoder.audio = (SimpleNamespace(**config.raw_encoder.audio)
                                if hasattr(config.raw_encoder, 'audio') and config.raw_encoder.audio
                                else None)

    config.dynamic_resolution = SimpleNamespace(**config.dynamic_resolution)
    config.output_convergence = SimpleNamespace(**config.output_convergence)
    config.training = SimpleNamespace(**config.training)

    train(config)
