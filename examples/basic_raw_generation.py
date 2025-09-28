# examples/basic_raw_generation.py
import subprocess
import os
import sys

# Ensure we are in the root directory
project_root = os.path.dirname(os.path.abspath(__file__)) + '/../'
os.chdir(project_root)

# --- Configuration ---
config_path = "src/config/model_config.yaml"
model_file = "aperture_llm_model_epoch_50.pt"  # Updated to reflect 50 epochs
seed_value = 42  # Consistent seed for reproducibility

# --- Prepare PYTHONPATH for subprocesses ---
# Add the project root to the PYTHONPATH. This makes 'src' discoverable as a top-level package.
current_python_path = os.environ.get('PYTHONPATH', '')
# Ensure project_root is at the beginning of PYTHONPATH
new_python_path = os.path.abspath(project_root) + os.pathsep + current_python_path
subprocess_env = os.environ.copy()
subprocess_env['PYTHONPATH'] = new_python_path

# --- Clean up previous model file if it exists ---
if os.path.exists(model_file):
    print(f"Removing old model file: {model_file}")
    os.remove(model_file)

print(f"--- Starting APERTURE-LLM Prototype Demonstration (Seed: {seed_value}) ---")

print("\n--- Training APERTURE-LLM Prototype (this will generate a model file) ---")
# Use sys.executable to ensure the correct Python interpreter is used (e.g., from venv)
train_command = [sys.executable, "src/scripts/train_model.py", "--config", config_path]
train_result = subprocess.run(train_command, capture_output=True, text=True, env=subprocess_env)
print(train_result.stdout)
if train_result.stderr:
    print("Training Errors:\n", train_result.stderr)
    if train_result.returncode != 0:
        print("Training failed. Exiting.")
        exit(1)

if not os.path.exists(model_file):
    print(f"Error: Model file '{model_file}' not found after training. Training may have failed.")
    exit(1)

print("\n--- Running Inference with low focus_strength (more exploratory) ---")
print("Expected: More varied, potentially less coherent output due to higher temperature/top_p.")
infer_command_low_focus = [
    sys.executable, "src/scripts/infer_model.py",  # Use sys.executable
    "--config", config_path,
    "--model_path", model_file,
    "--raw_text_input", "The future of AI is",
    "--focus_strength", "0.2",
    "--max_new_tokens", "100"  # Increased from 50
]
infer_result_low = subprocess.run(infer_command_low_focus, capture_output=True, text=True, env=subprocess_env)
print(infer_result_low.stdout)
if infer_result_low.stderr:
    print("Low Focus Inference Errors:\n", infer_result_low.stderr)

print("\n--- Running Inference with high focus_strength (more decisive) ---")
print("Expected: More repetitive or fixed output due to lower temperature/top_p, reflecting 'conceptual collapse'.")
infer_command_high_focus = [
    sys.executable, "src/scripts/infer_model.py",  # Use sys.executable
    "--config", config_path,
    "--model_path", model_file,
    "--raw_text_input", "The future of AI is",
    "--focus_strength", "0.9",
    "--max_new_tokens", "100"  # Increased from 50
]
infer_result_high = subprocess.run(infer_command_high_focus, capture_output=True, text=True, env=subprocess_env)
print(infer_result_high.stdout)
if infer_result_high.stderr:
    print("High Focus Inference Errors:\n", infer_result_high.stderr)

print("\n--- Running a placeholder evaluation ---")
eval_command = [
    sys.executable, "src/scripts/evaluate_model.py",
    "--config", config_path,
    "--model_path", model_file
]
eval_result = subprocess.run(eval_command, capture_output=True, text=True, env=subprocess_env)
print(eval_result.stdout)
if eval_result.stderr:
    print("Evaluation Errors:\n", eval_result.stderr)

print("\n--- APERTURE-LLM Prototype Demonstration Complete ---")
print("NOTE: The generated text will likely be simple and nonsensical in this prototype ")
print("due to the minimal model size and dummy training data. ")
print("The primary goal is to demonstrate the *execution flow* and *effect of focus_strength*.")
