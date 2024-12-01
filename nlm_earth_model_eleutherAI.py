from transformers import pipeline

# Load the pipeline
generator = generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=0)


# Generate text
result = generator(
    "How Earth came to be and we are all still here",
    max_length=300,
    num_return_sequences=2,
    temperature=0.7,  # Creativity level
    top_p=0.9        # Nucleus sampling
)
print(result)

