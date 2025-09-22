# examples/basic_raw_generation.py
import subprocess
import os
import shutil

# Ensure we are in the root directory
project_root = os.path.dirname(os.path.abspath(__file__)) + '/../'
os.chdir(project_root)

# --- Configuration ---
config_path = "src/config/model_config.yaml"
model_file = "aperture_llm_model_epoch_5.pt" # Renamed model file to save

# --- Clean up previous model file if it exists ---
if os.path.exists(model_file):
    print(f"Removing old model file: {model_file}")
    os.remove(model_file)

print("--- Training APERTURE-LLM Prototype (this will generate a model file) ---") # Renamed description
train_command = ["python", "src/scripts/train_model.py", "--config", config_path]
train_result = subprocess.run(train_command, capture_output=True, text=True)
print(train_result.stdout)
if train_result.stderr:
    print("Training Errors:\n", train_result.stderr)
    if train_result.returncode != 0:
        print("Training failed. Exiting.")
        exit(1)

if not os.path.exists(model_file):
    print(f"Error: Model file '{model_file}' not found after training. Training may have failed.")
    exit(1)

print(f"\n--- Running Inference with low focus_strength (more exploratory) ---")
infer_command_low_focus = [
    "python", "src/scripts/infer_model.py",
    "--config", config_path,
    "--model_path", model_file,
    "--raw_text_input", "The future of AI is",
    "--focus_strength", "0.2",
    "--max_new_tokens", "50"
]
infer_result_low = subprocess.run(infer_command_low_focus, capture_output=True, text=True)
print(infer_result_low.stdout)
if infer_result_low.stderr:
    print("Low Focus Inference Errors:\n", infer_result_low.stderr)

print(f"\n--- Running Inference with high focus_strength (more decisive) ---")
infer_command_high_focus = [
    "python", "src/scripts/infer_model.py",
    "--config", config_path,
    "--model_path", model_file,
    "--raw_text_input", "The future of AI is",
    "--focus_strength", "0.9",
    "--max_new_tokens", "50"
]
infer_result_high = subprocess.run(infer_command_high_focus, capture_output=True, text=True)
print(infer_result_high.stdout)
if infer_result_high.stderr:
    print("High Focus Inference Errors:\n", infer_result_high.stderr)

print(f"\n--- Running a placeholder evaluation ---")
eval_command = ["python", "src/scripts/evaluate_model.py", "--config", config_path, "--model_path", model_file]
eval_result = subprocess.run(eval_command, capture_output=True, text=True)
print(eval_result.stdout)
if eval_result.stderr:
    print("Evaluation Errors:\n", eval_result.stderr)

print("\n--- APERTURE-LLM Prototype Demonstration Complete ---") # Renamed description
print("NOTE: The generated text will likely be simple due to limited training data in this prototype.")
