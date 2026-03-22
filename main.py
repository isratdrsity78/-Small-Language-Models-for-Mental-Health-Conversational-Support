# Step 2: Import libraries and define paths
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from tqdm import tqdm

# NOTE: Unsloth requires a GPU (CUDA) to run. CPU-only execution is not supported.
# If you don't have a GPU, consider using Google Colab or other cloud platforms with free GPU access.

# Define the ID of the original base model you used for finetuning
MODEL_ID = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"

# Define the path to your downloaded and unzipped hf_model folder
# Make sure this path is correct on your local PC and the folder 'hf_model' exists here
model_path = "./hf_model"

max_seq_length = 2048  # Use the same max_seq_length as during training

# Load the base model. Unsloth will automatically download it if not present.
# The LoRA adapters from 'hf_model' will be loaded on top of this base model.
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=max_seq_length,
    dtype=None,  # Auto-detect dtype
    load_in_4bit=True,  # Load in 4-bit to save memory
)

# Load the tokenizer from your local hf_model folder to ensure any custom templates/tokens are applied
print("Loading tokenizer from local model folder...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Re-apply the chat template (important for correct inference formatting)
from unsloth.chat_templates import get_chat_template

print("Applying chat template...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3-thinking",
    mapping={
        "role": "role",
        "content": "content",
        "user": "user",
        "assistant": "assistant",
    },
)

print("Model and Tokenizer loaded successfully.")


# Step 3: Save the model as GGUF with f16 quantization
# This will create a file named 'model-f16.gguf' in the current directory
print("Saving model as GGUF with f16 quantization...")
model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
print("Model saved as GGUF with f16 quantization as 'model-f16.gguf'!")
