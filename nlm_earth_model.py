from transformers import pipeline

# Load the pipeline
generator = pipeline("text-generation", model="gpt2")

# Generate text
result = generator("How Earth came to be", max_length=500, num_return_sequences=2)
print(result)

