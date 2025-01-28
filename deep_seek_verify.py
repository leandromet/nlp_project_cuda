import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Set the cache directory (if needed)
# os.environ['TRANSFORMERS_CACHE'] = '/usr/local/huggingface_cache'

# Load the JSON configuration file
config_path = '/srv/nlp_project/config.json'
with open(config_path, 'r') as f:
    config_dict = json.load(f)

# Save the configuration dictionary to a temporary file
temp_config_path = '/srv/nlp_project/temp_config.json'
with open(temp_config_path, 'w') as f:
    json.dump(config_dict, f)

# Create the configuration object from the JSON file
config = AutoConfig.from_pretrained(temp_config_path)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with mixed precision (FP16)
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype=torch.float16, device_map="auto"
    )
except ValueError as e:
    print(f"Error loading model: {e}")
    print("Attempting to load model without quantization...")
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")

# Tokenize input
input_text = "What map layers can I use as a biodiversity researcher in Brazil?"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Move inputs to the same device as the model
device = next(model.parameters()).device
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate output
with torch.no_grad():
    output = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=400,  # Maximum number of tokens to generate
        num_return_sequences=1,  # Number of sequences to generate
        no_repeat_ngram_size=2,  # Prevent repetition of n-grams
        top_k=50,  # Top-k sampling
        top_p=0.95,  # Nucleus sampling
        temperature=0.7  # Sampling temperature
    )

# Decode the output tokens to text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(output_text)

# Clear GPU cache
torch.cuda.empty_cache()