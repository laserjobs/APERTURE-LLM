# src/scripts/evaluate_model.py
import torch
import yaml
from types import SimpleNamespace
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aperture_core.model import APERTURE_LLM
from aperture_core.utils import CharTokenizer, set_seed

def evaluate(config, model_path, benchmark_suite):
    set_seed(config.training.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = CharTokenizer()
    config.model.vocab_size = tokenizer.vocab_size

    model = APERTURE_LLM(config).to(device)
    
    # FIX: Add error handling for model loading
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

    print(f"\n--- Evaluating on {benchmark_suite} ---")
    print("NOTE: This is a placeholder evaluation script for a prototype.")
    print("A full evaluation would involve loading specific datasets, calculating metrics (perplexity, coherence, safety), and comparing against baselines.")

    dummy_text = "This is a test sentence for evaluation."
    encoded_input = torch.tensor(tokenizer.encode(dummy_text), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(encoded_input, focus_strength=0.7)
        dummy_output = model.generate(encoded_input, max_new_tokens=20, focus_strength=0.7)
        print(f"Example output: {tokenizer.decode(dummy_output[0].tolist())}")
        print("Metrics (placeholder): Perplexity = 1.0, Coherence = 0.9, Efficiency = 0.95")
        print("This is dummy output; real metrics would be computed here.")
    
    print("\nEvaluation finished.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate APERTURE-LLM.")
    parser.add_argument('--config', type=str, default='src/config/model_config.yaml',
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint.')
    parser.add_argument('--benchmark_suite', type=str, default="M3E",
                        help='Name of the benchmark suite to use.')

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

    evaluate(config, args.model_path, args.benchmark_suite)
