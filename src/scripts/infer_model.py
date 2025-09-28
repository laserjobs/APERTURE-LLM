import sys
import os
import warnings
import torch  # Moved up
import yaml  # Moved up
from types import SimpleNamespace  # Moved up

# Suppress FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

# Add src/aperture_core to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aperture_core.model import APERTURE_LLM
from aperture_core.utils import CharTokenizer, set_seed


# Placeholder function to load image (replace with actual image loading using torch operations)
def load_dummy_image(batch_size, device, config):
    # This should match the expected input shape for UniversalRawImageEncoder
    return torch.randn(batch_size, config.raw_encoder.image.input_shape[0],
                       config.raw_encoder.image.input_shape[1],
                       config.raw_encoder.image.input_shape[2], device=device)


# Placeholder function to load audio (replace with actual audio loading using torch operations)
def load_dummy_audio(batch_size, device, config):
    # This should match the expected input shape for UniversalRawAudioEncoder
    return torch.randn(batch_size, config.raw_encoder.audio.num_samples, device=device)


def infer(config, model_path, raw_text_input_str, focus_strength, max_new_tokens, output_modality,
          raw_image_input_arg=None, raw_audio_input_arg=None, targets_arg=None,
          adaptation_steps_limit=None):
    # C901: 'infer' is too complex (17). Consider refactoring into smaller functions.

    set_seed(config.training.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = CharTokenizer()
    config.model.vocab_size = tokenizer.vocab_size

    model = APERTURE_LLM(config).to(device)

    # Error handling for model loading
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint {model_path} not found.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: Failed to load model checkpoint {model_path}. Details: {e}")
        sys.exit(1)

    # 1. Prepare text input
    encoded_input_list = tokenizer.encode(raw_text_input_str)
    encoded_input = torch.tensor(encoded_input_list, dtype=torch.long, device=device)

    if encoded_input.size(0) == 0:
        print("Error: Input prompt is empty or contains no recognized characters.")
        sys.exit(1)

    # Truncate if input length exceeds model's block_size
    if encoded_input.size(0) > config.model.block_size:
        print(f"Warning: Input prompt length ({encoded_input.size(0)}) exceeds model's block_size "
              f"({config.model.block_size}). Truncating input.")
        encoded_input = encoded_input[-config.model.block_size:]

    encoded_input = encoded_input.unsqueeze(0)  # Add batch dimension

    # 2. Prepare multi-modal inputs
    raw_image_input_tensor = None
    if config.raw_encoder.image.enabled:
        if raw_image_input_arg == 'dummy':
            raw_image_input_tensor = load_dummy_image(1, device, config)  # Batch size 1 for inference
        elif raw_image_input_arg is not None:
            # Placeholder for loading actual image data
            print(f"Loading actual image from {raw_image_input_arg} (placeholder: using dummy data)")
            raw_image_input_tensor = load_dummy_image(1, device, config)

    raw_audio_input_tensor = None
    if config.raw_encoder.audio.enabled:
        if raw_audio_input_arg == 'dummy':
            raw_audio_input_tensor = load_dummy_audio(1, device, config)  # Batch size 1 for inference
        elif raw_audio_input_arg is not None:
            # Placeholder for loading actual audio data
            print(f"Loading actual audio from {raw_audio_input_arg} (placeholder: using dummy data)")
            raw_audio_input_tensor = load_dummy_audio(1, device, config)

    # 3. Prepare targets for online adaptation (if provided)
    targets_tensor = None
    if targets_arg:
        if targets_arg == 'dummy':
            # Create a dummy target sequence. It needs to be AT LEAST as long as prompt + max_new_tokens.
            # `encoded_input.size(1)` is the prompt length.
            required_target_len = encoded_input.size(1) + max_new_tokens

            dummy_target_str = raw_text_input_str + " "  # Start with prompt text
            # Append repeated text until it's long enough
            while len(tokenizer.encode(dummy_target_str)) < required_target_len:
                dummy_target_str += "the quick brown fox jumps over the lazy dog. "
                # Add a safety break to prevent excessively long strings or infinite loops
                if len(dummy_target_str) > required_target_len * 2:
                    break

            targets_list = tokenizer.encode(dummy_target_str)[:required_target_len]  # Truncate to exact required length
            targets_tensor = torch.tensor(targets_list, dtype=torch.long, device=device).unsqueeze(0)
            print(f"Using dummy targets for online adaptation (length: {targets_tensor.size(1)}).")
        else:
            # Placeholder for loading actual target data from file
            print(f"Loading actual targets from {targets_arg} (placeholder: using dummy data).")
            # Example: read file and encode
            # with open(targets_arg, 'r') as f:
            #     target_text = f.read()
            # targets_list = tokenizer.encode(target_text)

            # For this prototype, we'll still use dummy logic if a file path is given
            dummy_target_str = raw_text_input_str + " the quick brown fox jumps over the lazy dog."
            targets_list = tokenizer.encode(dummy_target_str)
            targets_tensor = torch.tensor(targets_list, dtype=torch.long, device=device).unsqueeze(0)


    # 4. Generate
    print(f"\n--- Generating with focus_strength={focus_strength:.2f} ---")
    generated_indices = model.generate(
        raw_text_input=encoded_input,
        max_new_tokens=max_new_tokens,
        focus_strength=focus_strength,
        raw_image_input=raw_image_input_tensor,
        raw_audio_input=raw_audio_input_tensor,
        targets=targets_tensor,  # Pass targets for online adaptation
        adaptation_steps_limit=adaptation_steps_limit  # Pass adaptation step limit
    )
    generated_text = tokenizer.decode(generated_indices[0].tolist())

    # 5. Output
    if output_modality == "text":
        print(f"Prompt: {raw_text_input_str}")
        print(f"Generated: {generated_text}")
    else:
        print(f"Generated output (raw indices): {generated_indices[0].tolist()}")
        print("NOTE: Multi-modal output generation is a future feature in this prototype.")


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
    parser.add_argument('--raw_image_input', type=str, default=None,
                        help="Path to raw image data, or 'dummy' to use random data.")
    parser.add_argument('--raw_audio_input', type=str, default=None,
                        help="Path to raw audio data, or 'dummy' to use random data.")
    parser.add_argument('--targets', type=str, default=None,
                        help="Path to target text for online adaptation, or 'dummy' to use dummy targets.")
    parser.add_argument('--adaptation_steps_limit', type=int, default=None,
                        help="Limit online adaptation to this many initial generation steps.")

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
    config.raw_encoder.image = (SimpleNamespace(**config.raw_encoder.image)
                                if hasattr(config.raw_encoder, 'image') and config.raw_encoder.image
                                else SimpleNamespace(enabled=False))
    config.raw_encoder.audio = (SimpleNamespace(**config.raw_encoder.audio)
                                if hasattr(config.raw_encoder, 'audio') and config.raw_encoder.audio
                                else SimpleNamespace(enabled=False))

    config.dynamic_resolution = SimpleNamespace(**config.dynamic_resolution)
    config.output_convergence = SimpleNamespace(**config.output_convergence)
    config.training = SimpleNamespace(**config.training)

    infer(config, args.model_path, args.raw_text_input, args.focus_strength, args.max_new_tokens, args.output_modality,
          args.raw_image_input, args.raw_audio_input, args.targets, args.adaptation_steps_limit)
