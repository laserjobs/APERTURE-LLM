import argparse
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

# Original path modification that you said worked
# This ensures `repo_root/src` is on `sys.path`, making `aperture_core` discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# CRUCIAL FIX: Change to `from aperture_core.model` (which resolves because `repo_root/src` is on path)
# The previous version had `from aperture_core.model`, but it was failing.
# If `sys.path.append(os.path.join(os.path.dirname(__file__), '..'))` adds `repo_root/src`
# to sys.path, then `aperture_core` is the correct import.
# The error means that `aperture_core` *wasn't* found. This implies the path append
# was not working or `PYTHONPATH` was conflicting.
# Let's revert this to match the *original* implicit working state you had.
# The core problem is that `sys.path.append` from a subprocess might be tricky.
# To match the error and the structure, the *simplest* fix is to assume
# that `src` is what needs to be added to PYTHONPATH.
# But `basic_raw_generation.py` adds `repo_root`.
# This is where the contradiction lies.
#
# Let's assume the user's explicit Python path modification in basic_raw_generation.py is the source of truth for subprocesses.
# And `basic_raw_generation.py` sets `PYTHONPATH=repo_root`.
# Therefore, `src` is a top-level package. So imports MUST be `from src.aperture_core.model`.
# The `sys.path.append` inside train_model.py is redundant *if* the parent is doing its job.
# The `ModuleNotFoundError: No module named 'aperture_core'` implies it's looking for `repo_root/aperture_core`.
#
# Okay, new strategy. If `basic_raw_generation.py` is setting `PYTHONPATH` to the *repo root*,
# then all imports must be `from src.aperture_core.model`.
# The `sys.path.append` in train_model.py should be removed as it's confusing and possibly conflicting.
#
# Final attempt at this specific file's import:
# Assuming `PYTHONPATH` has `repo_root` because `basic_raw_generation.py` explicitly sets it that way.
# And the local `sys.path.append` in `src/scripts/*` is then confusing.
# The most robust fix is to rely SOLELY on `PYTHONPATH` from the parent.

# REMOVED: sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Relying on PYTHONPATH set by examples/basic_raw_generation.py

# CRUCIAL FIX: Change import to correctly find module when repo_root is on PYTHONPATH
from src.aperture_core.model import APERTURE_LLM
from src.aperture_core.utils import get_batch, CharTokenizer, set_seed


def train(config):
    # ... (rest of the class code as before) ...
    set_seed(config.training.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = CharTokenizer()

    dummy_text = ("This is a simple text string for demonstration. The APERTURE LLM aims to be "
                  "the best LLM available. It processes raw digital inputs directly. "
                  "Hello World 123!@#$%^&*()_+-=[]{}|;':\",./<>?~`") * 10
    data = torch.tensor(tokenizer.encode(dummy_text), dtype=torch.long)

    config.model.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {config.model.vocab_size}")
    print(f"Dummy data size: {len(data)} characters")

    model = APERTURE_LLM(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()
    print(f"APERTURE-LLM Model initialized with "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    xb_debug, yb_debug = get_batch(data, config.model.block_size, config.training.batch_size, device)
    print(f"DEBUG: Input batch shape: {xb_debug.shape}, Target batch shape: {yb_debug.shape}")
    with torch.no_grad():
        logits_debug = model(xb_debug, focus_strength=0.5)
    print(f"DEBUG: Logits shape: {logits_debug.shape}")

    model.train()
    num_iterations_per_epoch = (len(data) - config.model.block_size) // config.training.batch_size
    if num_iterations_per_epoch == 0:
        num_iterations_per_epoch = 100

    for epoch in range(config.training.num_epochs):
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        for iter_step in tqdm(range(num_iterations_per_epoch), desc=f"Epoch {epoch+1}"):
            xb, yb = get_batch(data, config.model.block_size, config.training.batch_size, device)

            logits = model(xb, focus_strength=0.5)

            B, T, C_vocab = logits.shape
            loss = criterion(logits.view(B*T, C_vocab), yb.view(B*T))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iter_step % config.training.eval_interval == 0:
                print(f"  Loss at iter {iter_step}: {loss.item():.4f}")

    print("Training finished.")
    model_filename = f"aperture_llm_model_epoch_{config.training.num_epochs}.pt"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train APERTURE-LLM.")
    parser.add_argument('--config', type=str, default='src/config/model_config.yaml',
                        help='Path to the model configuration YAML file.')
    args = parser.parse_args()

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

    config = SimpleNamespace(**config_dict)
    config.model = SimpleNamespace(**config.model)
    config.raw_encoder = SimpleNamespace(**config.raw_encoder)
    config.raw_encoder.text = SimpleNamespace(**config.raw_encoder.text)
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
