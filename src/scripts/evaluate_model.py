import argparse
import os
import sys
import warnings
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml

# REMOVED: sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# CRUCIAL FIX: Change import to correctly find module when repo_root is on PYTHONPATH
from src.aperture_core.model import APERTURE_LLM
from src.aperture_core.utils import CharTokenizer, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)


def evaluate(config, model_path, benchmark_suite):
    # ... (function body is identical to previous version, only imports changed) ...
    set_seed(config.training.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = CharTokenizer()
    config.model.vocab_size = tokenizer.vocab_size

    model = APERTURE_LLM(config).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
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
    print("A full evaluation would involve loading specific datasets, calculating metrics "
          "(perplexity, coherence, safety), and comparing against baselines.")

    dummy_eval_text = "This is a test sentence for evaluation."

    encoded_full_input = torch.tensor(tokenizer.encode(dummy_eval_text),
                                      dtype=torch.long, device=device).unsqueeze(0)

    raw_image_input_eval = None
    if config.raw_encoder.image.enabled:
        raw_image_input_eval = torch.randn(1, config.raw_encoder.image.input_shape[0],
                                           config.raw_encoder.image.input_shape[1],
                                           config.raw_encoder.image.input_shape[2], device=device)

    raw_audio_input_eval = None
    if config.raw_encoder.audio.enabled:
        raw_audio_input_eval = torch.randn(1, config.raw_encoder.audio.num_samples, device=device)

    if encoded_full_input.size(1) < 2:
        print("Warning: Evaluation text too short to compute perplexity. Skipping perplexity calculation.")
        if encoded_full_input.size(1) == 0:
            print("No valid input characters for generation.")
            sys.exit(1)
        with torch.no_grad():
            dummy_output = model.generate(
                encoded_full_input, max_new_tokens=20, focus_strength=0.7,
                raw_image_input=raw_image_input_eval, raw_audio_input=raw_audio_input_eval
            )
            print(f"Example output (from short prompt): {tokenizer.decode(dummy_output[0].tolist())}")
        print("Metrics (placeholder): Coherence = 0.9, Efficiency = 0.95")
        print("\nEvaluation finished.")
        return

    input_for_loss = encoded_full_input[:, :-1]
    target_for_loss = encoded_full_input[:, 1:]

    with torch.no_grad():
        logits = model(
            input_for_loss,
            raw_image_input=raw_image_input_eval,
            raw_audio_input=raw_audio_input_eval,
            focus_strength=0.7
        )

        B, T_fused, C_vocab = logits.shape
        T_text = input_for_loss.size(1)

        if T_fused < T_text:
            print(f"Warning: Fused features length ({T_fused}) is less than expected text input "
                  f"length ({T_text}). Adjusting text_logits slice.")
            T_text = min(T_text, T_fused)

        text_logits = logits[:, :T_text, :]

        loss = F.cross_entropy(text_logits.view(B*T_text, C_vocab), target_for_loss.view(B*T_text))
        perplexity = torch.exp(loss)

        dummy_output = model.generate(
            encoded_full_input, max_new_tokens=20, focus_strength=0.7,
            raw_image_input=raw_image_input_eval, raw_audio_input=raw_audio_input_eval
        )
        print(f"Example output: {tokenizer.decode(dummy_output[0].tolist())}")
        print(f"Computed Perplexity: {perplexity.item():.4f}")
        print("Metrics (placeholder): Coherence = 0.9, Efficiency = 0.95")
        print("This is dummy output; real metrics would be computed here.")

    print("\nEvaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate APERTURE-LLM.")
    parser.add_argument('--config', type=str, default='src/config/model_config.yaml',
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint.')
    parser.add_argument('--benchmark_suite', type=str, default="M3E",
                        help='Name of the benchmark suite to use.')

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
                                else SimpleNamespace(enabled=False))
    config.raw_encoder.audio = (SimpleNamespace(**config.raw_encoder.audio)
                                if hasattr(config.raw_encoder, 'audio') and config.raw_encoder.audio
                                else SimpleNamespace(enabled=False))

    config.dynamic_resolution = SimpleNamespace(**config.dynamic_resolution)
    config.output_convergence = SimpleNamespace(**config.output_convergence)
    config.training = SimpleNamespace(**config.training)

    evaluate(config, args.model_path, args.benchmark_suite)
