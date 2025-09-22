# src/scripts/infer_model.py
import torch
import yaml
from types import SimpleNamespace
import sys
import os

# Add src/prometheus_core to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prometheus_core.model import Prometheus
from prometheus_core.utils import CharTokenizer

def infer(config, model_path, raw_text_input, focus_strength, max_new_tokens, output_modality):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load tokenizer and model
    tokenizer = CharTokenizer()
    config.model.vocab_size = tokenizer.vocab_size # Update vocab_size based on tokenizer
    
    model = Prometheus(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # 2. Prepare input
    encoded_input = torch.tensor(tokenizer.encode(raw_text_input), dtype=torch.long, device=device).unsqueeze(0) # Add batch dim

    # 3. Generate
    print(f"\n--- Generating with focus_strength={focus_strength:.2f} ---")
    # For prototype, only raw_text_input is handled for generation
    generated_indices = model.generate(encoded_input, max_new_tokens, focus_strength=focus_strength)
    generated_text = tokenizer.decode(generated_indices[0].tolist())

    # 4. Output
    if output_modality == "text":
        print(f"Prompt: {raw_text_input}")
        print(f"Generated: {generated_text}")
    else:
        print(f"Generated output (raw indices): {generated_indices[0].tolist()}")
        print(f"NOTE: Multi-modal output generation is a future feature in this prototype.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Infer with Prometheus LLM.")
    parser.add_argument('--config', type=str, default='src/config/model_config.yaml',
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint.')
    parser.add_argument('--raw_text_input', type=str, default="The nature of consciousness is",
                        help='Raw text prompt for generation.')
    parser.add_argument('--focus_strength', type=float, default=0.5,
                        help='Focus strength for non-linear output convergence (0.0 to 1.0).')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--output_modality', type=str, default="text",
                        help='Desired output modality (e.g., "text", "image", "audio").')

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
    config.training = SimpleNamespace(**config.training) # Not strictly needed for infer but good for consistency

    infer(config, args.model_path, args.raw_text_input, args.focus_strength, args.max_new_tokens, args.output_modality)
