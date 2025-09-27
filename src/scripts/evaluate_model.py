# src/scripts/evaluate_model.py
import torch
import torch.nn.functional as F # Added for perplexity calculation
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

    print(f"\n--- Evaluating on {benchmark_suite} ---")
    print("NOTE: This is a placeholder evaluation script for a prototype.")
    print("A full evaluation would involve loading specific datasets, calculating metrics (perplexity, coherence, safety), and comparing against baselines.")

    dummy_eval_text = "This is a test sentence for evaluation." # Changed var name to avoid conflict if `encoded_input` was re-used in another part of the script for generation
    
    # Encode for loss calculation (targets are shifted input)
    # The input for evaluating perplexity should align with how the model was trained (predicting the next character).
    # We take the actual text, then compute loss of model's prediction of text[1:] given text[:-1]
    encoded_full_input = torch.tensor(tokenizer.encode(dummy_eval_text), dtype=torch.long, device=device).unsqueeze(0)
    
    # Ensure encoded_full_input is not empty and has a sequence length suitable for evaluation
    if encoded_full_input.size(1) < 2:
        print("Warning: Evaluation text too short to compute perplexity.")
        # Fallback to just generation if perplexity cannot be computed
        if encoded_full_input.size(1) == 0:
            print("No valid input characters for generation.")
            sys.exit(1)
        # If one character, still try to generate
        with torch.no_grad():
            dummy_output = model.generate(encoded_full_input, max_new_tokens=20, focus_strength=0.7)
            print(f"Example output (from short prompt): {tokenizer.decode(dummy_output[0].tolist())}")
        print("Metrics (placeholder): Coherence = 0.9, Efficiency = 0.95")
        print("\nEvaluation finished.")
        return

    # Use the portion of the sequence for which we have both input and a target
    input_for_loss = encoded_full_input[:, :-1]
    target_for_loss = encoded_full_input[:, 1:]

    with torch.no_grad():
        # Get logits for the input_for_loss sequence
        logits = model(input_for_loss, focus_strength=0.7)
        
        # Calculate loss (and perplexity)
        # Reshape logits to (N*T, V) and targets to (N*T) for CrossEntropyLoss
        B, T, C_vocab = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C_vocab), target_for_loss.view(B*T))
        perplexity = torch.exp(loss)

        # For simplicity, we also print dummy output (this can use the full initial text or part of it)
        dummy_output = model.generate(encoded_full_input, max_new_tokens=20, focus_strength=0.7)
        print(f"Example output: {tokenizer.decode(dummy_output[0].tolist())}")
        print(f"Computed Perplexity: {perplexity.item():.4f}") # Display computed perplexity
        print("Metrics (placeholder): Coherence = 0.9, Efficiency = 0.95")
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
