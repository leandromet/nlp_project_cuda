from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Function to log GPU memory usage
def log_gpu_memory():
    if torch.cuda.is_available():
        print("\n[GPU MEMORY USAGE]")
        print(torch.cuda.memory_summary(device=0, abbreviated=False))
    else:
        print("\n[INFO] GPU not available. Using CPU.")

# Log the start time
start_time = time.time()

# Log: Initializing the pipeline
print("[INFO] Initializing the model and pipeline...")

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Log: Loaded model details
log_gpu_memory()

# Tokenization check
input_text = "Are forests in danger from global warming? And are they responsible for it?"
tokens = tokenizer(input_text, return_tensors="pt")
print(f"\n[DEBUG] Input text: {input_text}")
print(f"[DEBUG] Tokenized input: {tokens}")

# Generate text
print("[INFO] Generating text...")
try:
    result = generator(
        input_text,
        max_length=100,
        num_return_sequences=2,
        temperature=0.7,  # Creativity level
        top_p=0.9        # Nucleus sampling
    )
    log_gpu_memory()
except RuntimeError as e:
    print("[ERROR] A RuntimeError occurred during generation.")
    print(str(e))
    torch.cuda.empty_cache()
    print("[INFO] Cleared GPU memory. Try reducing `max_length` or batch size.")

# Print results
print("\n[RESULTS]")
for i, res in enumerate(result):
    print(f"Sequence {i + 1}:\n{res['generated_text']}\n")

# Log total time taken
end_time = time.time()
print(f"[INFO] Total time elapsed: {end_time - start_time:.2f} seconds")
