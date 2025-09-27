# src/scripts/infer_model.py
import torch
import yaml
from types import SimpleNamespace
import sys
import os

# Add src/aperture_core to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aperture_core.model import APERTURE_LLM
from aperture_core.utils import CharTokenizer, set_seed

def infer(config, model_path, raw_text_input, focus_strength, max_new_tokens, output_modality):
    set_seed(config.training.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = CharTokenizer()
    config.model.vocab_size = tokenizer.vocab_size
    
    model = APERTURE_LLM(config).to(device)
    
    # Error handling for model loading
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint {model_path} not found.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: Failed to load model checkpoint {model_path}. Details: {e}")
        sys.exit(1)

    model.eval()

    # FIX: Truncate or pad the input prompt to block_size
    encoded_input_list = tokenizer.encode(raw_text_input)
    encoded_input = torch.tensor(encoded_input_list, dtype=torch.long, device=device)
    
    if encoded_input.size(0) == 0:
        print("Error: Input prompt is empty or contains no recognized characters.")
        sys.exit(1)

    if encoded_input.size(0) > config.model.block_size:
        print(f"Warning: Input prompt length ({encoded_input.size(0)}) exceeds model's block_size ({config.model.block_size}). Truncating input.")
        encoded_input = encoded_input[-config.model.block_size:]  # Truncate to block_size
    
    encoded_input = encoded_input.unsqueeze(0)  # Add batch dimension

    print(f"\n--- Generating with focus_strength={focus_strength:.2f} ---")
    generated_indices = model.generate(encoded_input, max_new_tokens, focus_strength=focus_strength)
    generated_text = tokenizer.decode(generated_indices[0].tolist())

    if output_modality == "text":
        print(f"Prompt: {raw_text_input}")
        print(f"Generated: {generated_text}")
    else:
        print(f"Generated output (raw indices): {generated_indices[0].tolist()}")
        print(f"NOTE: Multi-modal output generation is a future feature in this prototype.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Infer with APERTURE-LLM.")
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

    # Error handling for config loading
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
    config.raw_encoder.image = SimpleNamespace(**config.raw_encoder.image) if hasattr(config.raw_encoder, 'image') and config.raw_encoder.image else None
    config.raw_encoder.audio = SimpleNamespace(**config.raw_encoder.audio) if hasattr(config.raw_encoder, 'audio') and config.raw_encoder.audio else None

    config.dynamic_resolution = SimpleNamespace(**config.dynamic_resolution)
    config.output_convergence = SimpleNamespace(**config.output_convergence)
    config.training = SimpleNamespace(**config.training)

    infer(config, args.model_path, args.raw_text_input, args.focus_strength, args.max_new_tokens, args.output_modality)
