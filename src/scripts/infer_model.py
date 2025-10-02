import argparse
import os
import sys
import torch
from types import SimpleNamespace
import yaml
import warnings

# Now, import modules using the `src.aperture_core` prefix
from src.aperture_core.model import APERTURE_LLM
from src.aperture_core.utils import CharTokenizer, set_seed
from src.aperture_core.token_bridge import DummyExternalTokenizer, TokenToRawCharAdapter, RawCharToTokenAdapter

# Suppress FutureWarning from torch.load if weights_only=False is used on newer PyTorch versions
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
seed_value = 42


def load_config(config_path):
    """Loads and parses the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_path}. Details: {e}")
        sys.exit(1)

    # Convert dictionary to SimpleNamespace for easy attribute access
    def convert_dict_to_namespace(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: convert_dict_to_namespace(v) for k, v in obj.items()})
        return obj

    return convert_dict_to_namespace(config_dict)


def _setup_config(config_dict):
    """Converts configuration dictionary to SimpleNamespace structure robustly."""
    # Helper for robustly converting nested dicts to namespaces
    def to_ns_recursive(obj):
        if isinstance(obj, dict):
            ns = SimpleNamespace(**{k: to_ns_recursive(v) for k, v in obj.items()})
            return ns
        return obj

    # Apply recursive conversion to the whole config structure
    config = to_ns_recursive(config_dict)

    # Explicitly ensure modality encoders are present if they were missing in YAML
    if hasattr(config, 'raw_encoder') and isinstance(config.raw_encoder, SimpleNamespace):
        if not hasattr(config.raw_encoder, 'image'):
            config.raw_encoder.image = SimpleNamespace(enabled=False)
        if not hasattr(config.raw_encoder, 'audio'):
            config.raw_encoder.audio = SimpleNamespace(enabled=False)
    else:
        # Fallback if raw_encoder structure is completely missing/malformed
        config.raw_encoder = SimpleNamespace(enabled=False)

    return config


def _initialize_bridge(external_tokenizer, aperture_char_tokenizer, device):
    """Initializes Token and Raw Char Adapters."""
    token_to_raw_char_adapter = TokenToRawCharAdapter(external_tokenizer, aperture_char_tokenizer).to(device)
    raw_char_to_token_adapter = RawCharToTokenAdapter(external_tokenizer, aperture_char_tokenizer).to(device)
    print("Aperture-Token Bridge Adapters initialized.")
    return token_to_raw_char_adapter, raw_char_to_token_adapter


def main():
    parser = argparse.ArgumentParser(description="Demonstrate Aperture-Token Bridge.")
    parser.add_argument('--config', type=str, default='src/config/model_config.yaml',
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint.')

    # --- ADDITIONS: Arguments required for generation and multi-modal inference ---
    parser.add_argument('--raw_text_input', type=str, default="The future of AI is",
                        help='Raw text prompt for generation.')
    parser.add_argument('--max_new_tokens', type=int, default=20,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--focus_strength', type=float, default=0.5,
                        help='Focus strength for non-linear output convergence '
                             '(0.0 to 1.0).')
    parser.add_argument('--output_modality', type=str, default="text",
                        help='Desired output modality (e.g., "text", "image", "audio").')
    parser.add_argument('--raw_image_input', type=str, default=None,
                        help="Path to raw image data, or 'dummy' to use random data.")
    parser.add_argument('--raw_audio_input', type=str, default=None,
                        help="Path to raw audio data, or 'dummy' to use random data.")
    parser.add_argument('--targets', type=str, default=None,
                        help="Path to target text for online adaptation, or "
                             "'dummy' to use dummy targets.")
    parser.add_argument('--adaptation_steps_limit', type=int, default=None,
                        help="Limit online adaptation to this many initial "
                             "generation steps.")
    # --- END ADDITIONS ---

    # Original argument kept for reference, though not used in the failing command
    parser.add_argument('--benchmark_suite', type=str, default="M3E",
                        help='Name of the benchmark suite to use.')

    args = parser.parse_args()

    set_seed(seed_value)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Load APERTURE-LLM Config and Model ---

    # Load raw dictionary first
    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    config = _setup_config(config_dict)

    aperture_char_tokenizer = CharTokenizer()
    config.model.vocab_size = aperture_char_tokenizer.vocab_size

    aperture_model = APERTURE_LLM(config).to(device)

    # Load the trained APERTURE-LLM model
    if os.path.exists(args.model_path):
        try:
            aperture_model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
            print(f"APERTURE-LLM loaded successfully from {args.model_path}")
        except RuntimeError as e:
            print(f"Error: Failed to load APERTURE-LLM checkpoint {args.model_path}. Details: {e}")
            sys.exit(1)
    else:
        print(f"Warning: APERTURE-LLM checkpoint {args.model_path} not found. "
              "Please ensure training was successful. "
              "Proceeding with an untrained model for adapter testing, output may be random.")

    aperture_model.eval()  # Set model to evaluation mode

    # --- 2. Initialize Dummy External Tokenizer (representing a tokenized AI's tokenizer) ---
    external_tokenizer = DummyExternalTokenizer()
    print(f"Dummy External Tokenizer vocab size: {external_tokenizer.vocab_size}")

    # --- 3. Initialize Adapters ---
    token_to_raw_char_adapter, raw_char_to_token_adapter = _initialize_bridge(
        external_tokenizer, aperture_char_tokenizer, device
    )


    # --- Scenario 1: Tokenized AI output -> APERTURE-LLM Input ---
    print("\n--- Scenario 1: Tokenized AI Output (Tokens) -> APERTURE-LLM Input (Raw Chars) ---")
    tokenized_ai_output_text = "The quick brown fox jumps over the lazy dog."
    tokenized_ai_output_tokens = torch.tensor(external_tokenizer.encode(tokenized_ai_output_text),
                                              dtype=torch.long, device=device).unsqueeze(0)
    print(f"Tokenized AI Output (original text): '{tokenized_ai_output_text}'")
    print(f"Tokenized AI Output (encoded tokens): {tokenized_ai_output_tokens.tolist()}")

    aperture_input_chars = token_to_raw_char_adapter(tokenized_ai_output_tokens)
    print(f"APERTURE-LLM Input (raw chars, decoded): "
          f"'{aperture_char_tokenizer.decode(aperture_input_chars[0].tolist())}'")

    with torch.no_grad():
        aperture_processed_logits = aperture_model(aperture_input_chars, focus_strength=0.7)
        print(f"APERTURE-LLM processed the raw char input. Logits shape: "
              f"{aperture_processed_logits.shape}")

    # --- Scenario 2: APERTURE-LLM Output (Raw Chars) -> Tokenized AI Input (Tokens) ---
    print("\n--- Scenario 2: APERTURE-LLM Output (Raw Chars) -> Tokenized AI Input (Tokens) ---")
    aperture_prompt_text = args.raw_text_input
    aperture_prompt_chars = torch.tensor(aperture_char_tokenizer.encode(aperture_prompt_text),
                                         dtype=torch.long, device=device).unsqueeze(0)
    print(f"APERTURE-LLM Prompt (text): '{aperture_prompt_text}'")

    # Prepare dummy multimodal inputs if enabled, for generation testing
    raw_image_input_gen = None
    raw_audio_input_gen = None

    # Check configuration structure before accessing nested attributes for dummy inputs
    image_enabled = (hasattr(config, 'raw_encoder') and
                     hasattr(config.raw_encoder, 'image') and
                     config.raw_encoder.image.enabled)
    audio_enabled = (hasattr(config, 'raw_encoder') and
                     hasattr(config.raw_encoder, 'audio') and
                     config.raw_encoder.audio.enabled)

    if image_enabled and args.raw_image_input == 'dummy':
        # Load dummy image matching expected config shape (assuming batch size 1)
        if (hasattr(config.raw_encoder.image, 'input_shape') and
                len(config.raw_encoder.image.input_shape) == 3):
            raw_image_input_gen = torch.randn(1, config.raw_encoder.image.input_shape[0],
                                              config.raw_encoder.image.input_shape[1],
                                              config.raw_encoder.image.input_shape[2], device=device)

    if audio_enabled and args.raw_audio_input == 'dummy':
        # Requires config.raw_encoder.audio.num_samples to be defined if enabled
        if hasattr(config.raw_encoder.audio, 'num_samples'):
            raw_audio_input_gen = torch.randn(1, config.raw_encoder.audio.num_samples, device=device)


    with torch.no_grad():
        generated_indices = aperture_model.generate(
            raw_text_input=aperture_prompt_chars,
            max_new_tokens=args.max_new_tokens,
            focus_strength=args.focus_strength,
            raw_image_input=raw_image_input_gen,
            raw_audio_input=raw_audio_input_gen
        )

    aperture_generated_text = aperture_char_tokenizer.decode(generated_indices[0].tolist())
    print(f"APERTURE-LLM Generated (raw chars): '{aperture_generated_text}'")

    tokenized_ai_input_tokens = raw_char_to_token_adapter(generated_indices)
    print(f"Tokenized AI Input (tokens): {tokenized_ai_input_tokens.tolist()}")
    print(f"Tokenized AI Input (decoded): "
          f"'{external_tokenizer.decode(tokenized_ai_input_tokens[0].tolist())}'")
    print("Tokenized AI can now consume this output, benefiting from APERTURE-LLM's "
          "raw generation capabilities.")

    # --- Scenario 3: Mixed Multi-Modal Input to APERTURE-LLM, then text output to Tokenized AI ---
    print("\n--- Scenario 3: Multi-Modal Input to APERTURE-LLM -> Raw Chars -> Tokenized AI Input ---")

    if raw_image_input_gen is not None or raw_audio_input_gen is not None:
        aperture_mm_prompt_text = "Describe this scene:"
        aperture_mm_prompt_chars = torch.tensor(aperture_char_tokenizer.encode(aperture_mm_prompt_text),
                                                dtype=torch.long, device=device).unsqueeze(0)

        print(f"APERTURE-LLM Multi-Modal Prompt (text): '{aperture_mm_prompt_text}' "
              "with dummy image/audio.")

        with torch.no_grad():
            aperture_mm_generated_chars = aperture_model.generate(
                raw_text_input=aperture_mm_prompt_chars,
                max_new_tokens=70,
                focus_strength=0.8,
                raw_image_input=raw_image_input_gen,
                raw_audio_input=raw_audio_input_gen
            )
        aperture_mm_generated_text = aperture_char_tokenizer.decode(aperture_mm_generated_chars[0].tolist())
        print(f"APERTURE-LLM Multi-Modal Generated (raw chars): '{aperture_mm_generated_text}'")

        tokenized_ai_mm_input_tokens = raw_char_to_token_adapter(aperture_mm_generated_chars)
        print(f"Tokenized AI (consuming MM output) Decoded: "
              f"'{external_tokenizer.decode(tokenized_ai_mm_input_tokens[0].tolist())}'")
        print("Tokenized AI can now interpret APERTURE-LLM's multi-modal insights.")
    else:
        print("\nScenario 3 skipped: Multi-modal inputs were not explicitly requested/simulated.")

    print("\n--- Aperture-Token Bridge Demonstration Complete ---")


if __name__ == "__main__":
    main()
