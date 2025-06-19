# NLP Project CUDA

This project combines natural language processing (NLP) and geospatial processing with CUDA acceleration.

## Environment Setup

There are three ways to set up the environment:

### Option 1: Using the setup script

```bash
# Make the setup script executable
chmod +x setup_environment.sh

# Run the setup script
./setup_environment.sh
```

### Option 2: Manual installation with pip

```bash
# Create virtual environment
python3 -m venv env

# Activate virtual environment
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Option 3: Install specific packages

```bash
# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install core packages
pip install numpy matplotlib tqdm zarr psutil plotly kaleido geopandas

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install geospatial packages
pip install rasterio

# Install transformer-based packages
pip install transformers
```

## Verifying CUDA Support

After installation, you can verify that CUDA is working properly:

```bash
# Activate the environment
source env/bin/activate

# Run the CUDA test script
python test_cuda.py
```

You should see output confirming CUDA is available and showing your GPU information.

## Project Components

- **Geospatial Processing**: The `raster_proc` directory contains modules for processing geospatial data
- **NLP Models**: Various Python files for working with transformer models
- **Utilities**: Tools for checking system configuration and CUDA availability

## Usage Examples

### Testing PyTorch CUDA Support

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Using Transformer Models

```python
from transformers import pipeline

# Load the model on GPU (device 0)
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=0)

# Generate text
result = generator(
    "How Earth came to be",
    max_length=300,
    num_return_sequences=1
)
print(result)
```

### Processing Raster Data

See the `raster_proc/mpi_polygon.py` file for examples of how to process geospatial data with parallel processing.
