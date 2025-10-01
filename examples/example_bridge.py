# examples/example_bridge.py
import os
import sys
import torch
from types import SimpleNamespace
import yaml

# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.aperture_core.model import APERTURE_LLM
from src.aperture_core.utils import CharTokenizer, set_seed
from src.aperture_core.token_bridge import DummyExternalTokenizer, TokenToRawCharAdapter, RawCharToTokenAdapter

# --- Configuration ---
config_path = "src/config/model_config.yaml"
# This model file is expected to be created by src/scripts/train_model.py
model_file = "aperture_llm_model_epoch_1.pt" 
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
    config = SimpleNamespace(**config_dict)
    
    # Recursively convert nested dictionaries to SimpleNamespace
    def convert_dict_to_namespace(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: convert_dict_to_namespace(v) for k, v in obj.items()})
        return obj

    return convert_dict_to_namespace(config_dict)

def main():
    set_seed(seed_value)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Load APERTURE-LLM Config and Model ---
    aperture_config = load_config(config_path)
    aperture_char_tokenizer = CharTokenizer()
    # Update vocab_size in config based on actual tokenizer vocab size
    aperture_config.model.vocab_size = aperture_char_tokenizer.vocab_size

    aperture_model = APERTURE_LLM(aperture_config).to(device)
    
    # Load the trained APERTURE-LLM model
    if os.path.exists(model_file):
        try:
            aperture_model.load_state_dict(torch.load(model_file, map_location=device, weights_only=False))
            print(f"APERTURE-LLM loaded successfully from {model_file}")
        except RuntimeError as e:
            print(f"Error: Failed to load APERTURE-LLM checkpoint {model_file}. Details: {e}")
            sys.exit(1)
    else:
        print(f"Warning: APERTURE-LLM checkpoint {model_file} not found. "
              "Please run 'src/scripts/train_model.py' first to train the model. "
              "Proceeding with an untrained model for adapter testing, output may be random.")
    
    aperture_model.eval() # Set model to evaluation mode

    # --- 2. Initialize Dummy External Tokenizer (representing a tokenized AI's tokenizer) ---
    external_tokenizer = DummyExternalTokenizer()
    print(f"Dummy External Tokenizer vocab size: {external_tokenizer.vocab_size}")

    # --- 3. Initialize Adapters ---
    token_to_raw_char_adapter = TokenToRawCharAdapter(external_tokenizer, aperture_char_tokenizer).to(device)
    raw_char_to_token_adapter = RawCharToTokenAdapter(external_tokenizer, aperture_char_tokenizer).to(device)
    print("Aperture-Token Bridge Adapters initialized.")

    # --- Scenario 1: Tokenized AI output -> APERTURE-LLM Input ---
    print("\n--- Scenario 1: Tokenized AI Output (Tokens) -> APERTURE-LLM Input (Raw Chars) ---")
    tokenized_ai_output_text = "The quick brown fox jumps over the lazy dog."
    # Simulate a tokenized AI encoding its output
    tokenized_ai_output_tokens = torch.tensor(external_tokenizer.encode(tokenized_ai_output_text), dtype=torch.long, device=device).unsqueeze(0)
    print(f"Tokenized AI Output (original text): '{tokenized_ai_output_text}'")
    print(f"Tokenized AI Output (encoded tokens): {tokenized_ai_output_tokens.tolist()}")

    # Convert tokenized output to raw characters for APERTURE-LLM
    aperture_input_chars = token_to_raw_char_adapter(tokenized_ai_output_tokens)
    print(f"APERTURE-LLM Input (raw chars, decoded): '{aperture_char_tokenizer.decode(aperture_input_chars[0].tolist())}'")

    # Now, APERTURE-LLM can process this as raw input
    with torch.no_grad():
        # APERTURE-LLM's forward pass processes the raw character input
        aperture_processed_logits = aperture_model(aperture_input_chars, focus_strength=0.7)
        print(f"APERTURE-LLM processed the raw char input. Logits shape: {aperture_processed_logits.shape}")
        # Note: 'aperture_processed_logits' would then be used for APERTURE-LLM's own generation if desired

    # --- Scenario 2: APERTURE-LLM Output (Raw Chars) -> Tokenized AI Input (Tokens) ---
    print("\n--- Scenario 2: APERTURE-LLM Output (Raw Chars) -> Tokenized AI Input (Tokens) ---")
    aperture_prompt_text = "The future of AI is"
    # APERTURE-LLM's input is always raw characters
    aperture_prompt_chars = torch.tensor(aperture_char_tokenizer.encode(aperture_prompt_text), dtype=torch.long, device=device).unsqueeze(0)
    print(f"APERTURE-LLM Prompt (text): '{aperture_prompt_text}'")

    # APERTURE-LLM generates raw characters
    with torch.no_grad():
        aperture_generated_chars = aperture_model.generate(
            raw_text_input=aperture_prompt_chars,
            max_new_tokens=50,
            focus_strength=0.9 # High focus for more decisive output
        )
    aperture_generated_text = aperture_char_tokenizer.decode(aperture_generated_chars[0].tolist())
    print(f"APERTURE-LLM Generated (raw chars): '{aperture_generated_text}'")

    # Convert APERTURE-LLM's raw character output to tokens for a tokenized AI
    tokenized_ai_input_tokens = raw_char_to_token_adapter(aperture_generated_chars)
    print(f"Tokenized AI Input (tokens): {tokenized_ai_input_tokens.tolist()}")
    print(f"Tokenized AI Input (decoded): '{external_tokenizer.decode(tokenized_ai_input_tokens[0].tolist())}'")
    print("Tokenized AI can now consume this output, benefiting from APERTURE-LLM's raw generation capabilities.")

    # --- Scenario 3: Mixed Multi-Modal Input to APERTURE-LLM, then text output to Tokenized AI ---
    print("\n--- Scenario 3: Multi-Modal Input to APERTURE-LLM -> Raw Chars -> Tokenized AI Input ---")
    if aperture_config.raw_encoder.image.enabled and aperture_config.raw_encoder.audio.enabled:
        aperture_mm_prompt_text = "Describe this scene:"
        aperture_mm_prompt_chars = torch.tensor(aperture_char_tokenizer.encode(aperture_mm_prompt_text), dtype=torch.long, device=device).unsqueeze(0)

        # Dummy multi-modal inputs (replace with actual loaded data in a real application)
        raw_image_input = torch.randn(1, aperture_config.raw_encoder.image.input_shape[0],
                                      aperture_config.raw_encoder.image.input_shape[1],
                                      aperture_config.raw_encoder.image.input_shape[2], device=device)
        raw_audio_input = torch.randn(1, aperture_config.raw_encoder.audio.num_samples, device=device)

        print(f"APERTURE-LLM Multi-Modal Prompt (text): '{aperture_mm_prompt_text}' with dummy image/audio.")

        with torch.no_grad():
            aperture_mm_generated_chars = aperture_model.generate(
                raw_text_input=aperture_mm_prompt_chars,
                max_new_tokens=70,
                focus_strength=0.8,
                raw_image_input=raw_image_input,
                raw_audio_input=raw_audio_input
            )
        aperture_mm_generated_text = aperture_char_tokenizer.decode(aperture_mm_generated_chars[0].tolist())
        print(f"APERTURE-LLM Multi-Modal Generated (raw chars): '{aperture_mm_generated_text}'")

        # Convert to tokens for consumption by a tokenized AI
        tokenized_ai_mm_input_tokens = raw_char_to_token_adapter(aperture_mm_generated_chars)
        print(f"Tokenized AI (consuming MM output) Decoded: '{external_tokenizer.decode(tokenized_ai_mm_input_tokens[0].tolist())}'")
        print("Tokenized AI can now interpret APERTURE-LLM's multi-modal insights.")
    else:
        print("\nScenario 3 skipped: Multi-modal encoders are not enabled in model_config.yaml.")

    print("\n--- Aperture-Token Bridge Demonstration Complete ---")


if __name__ == "__main__":
    main()
