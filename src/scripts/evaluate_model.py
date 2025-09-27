# src/scripts/evaluate_model.py
import torch
import torch.nn.functional as F
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

    dummy_eval_text = "This is a test sentence for evaluation."
    
    # Encode for loss calculation (targets are shifted input)
    encoded_full_input = torch.tensor(tokenizer.encode(dummy_eval_text), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate dummy multi-modal inputs for evaluation if enabled
    raw_image_input_eval = None
    if config.raw_encoder.image.enabled:
        # Use batch size 1 for evaluation
        raw_image_input_eval = torch.randn(1, 3, 224, 224, device=device)
    
    raw_audio_input_eval = None
    if config.raw_encoder.audio.enabled:
        # Use batch size 1 for evaluation
        raw_audio_input_eval = torch.randn(1, 131072, device=device)

    # Ensure encoded_full_input is not empty and has a sequence length suitable for perplexity
    if encoded_full_input.size(1) < 2:
        print("Warning: Evaluation text too short to compute perplexity.")
        if encoded_full_input.size(1) == 0:
            print("No valid input characters for generation.")
            sys.exit(1)
        # If one character, still try to generate
        with torch.no_grad():
            dummy_output = model.generate(
                encoded_full_input, max_new_tokens=20, focus_strength=0.7,
                raw_image_input=raw_image_input_eval, raw_audio_input=raw_audio_input_eval
            )
            print(f"Example output (from short prompt): {tokenizer.decode(dummy_output[0].tolist())}")
        print("Metrics (placeholder): Coherence = 0.9, Efficiency = 0.95")
        print("\nEvaluation finished.")
        return

    # Use the portion of the sequence for which we have both input and a target
    input_for_loss = encoded_full_input[:, :-1]
    target_for_loss = encoded_full_input[:, 1:]

    with torch.no_grad():
        # Get logits for the input_for_loss sequence (now potentially multi-modal conditioned)
        logits = model(
            input_for_loss, 
            raw_image_input=raw_image_input_eval, 
            raw_audio_input=raw_audio_input_eval, 
            focus_strength=0.7
        )
        
        # Calculate loss (and perplexity) - IMPORTANT: slice logits to match text length
        B, T_fused, C_vocab = logits.shape
        T_text = input_for_loss.size(1) # Text sequence length for loss calculation
        
        # Ensure T_fused is at least T_text
        if T_fused < T_text:
             print(f"Warning: Fused features length ({T_fused}) < text input length ({T_text}). Loss calculation may be incorrect.")
             # Fallback to a partial loss or exit if this becomes an issue.
             # For now, we'll try to slice as much as possible, but it indicates a data mismatch.
             T_text = min(T_text, T_fused)

        text_logits = logits[:, :T_text, :] # (B, T_text, C_vocab)

        loss = F.cross_entropy(text_logits.view(B*T_text, C_vocab), target_for_loss.view(B*T_text))
        perplexity = torch.exp(loss)

        # For simplicity, we also print dummy output (this can use the full initial text or part of it)
        dummy_output = model.generate(
            encoded_full_input, max_new_tokens=20, focus_strength=0.7,
            raw_image_input=raw_image_input_eval, raw_audio_input=raw_audio_input_eval
        )
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
    # Ensure image/audio config sections are created as SimpleNamespace if they exist and are enabled
    # Otherwise, pass a SimpleNamespace with enabled=False to avoid errors
    config.raw_encoder.text = SimpleNamespace(**config.raw_encoder.text)
    config.raw_encoder.image = SimpleNamespace(**config.raw_encoder.image) if hasattr(config.raw_encoder, 'image') and config.raw_encoder.image else SimpleNamespace(enabled=False)
    config.raw_encoder.audio = SimpleNamespace(**config.raw_encoder.audio) if hasattr(config.raw_encoder, 'audio') and config.raw_encoder.audio else SimpleNamespace(enabled=False)

    config.dynamic_resolution = SimpleNamespace(**config.dynamic_resolution)
    config.output_convergence = SimpleNamespace(**config.output_convergence)
    config.training = SimpleNamespace(**config.training)

    evaluate(config, args.model_path, args.benchmark_suite)
