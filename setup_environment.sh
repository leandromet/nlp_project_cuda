#!/bin/bash
# Setup script for NLP Project CUDA environment
# Created on: May 23, 2025

echo "Creating Python virtual environment..."
python3 -m venv env

echo "Activating virtual environment..."
source env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing base dependencies..."
pip install numpy matplotlib tqdm zarr psutil plotly kaleido geopandas squarify

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing geospatial dependencies..."
pip install rasterio

echo "Installing transformer-based dependencies..."
pip install transformers

echo "Installing Jupyter notebook dependencies..."
pip install jupyter notebook

echo "Installing development tools..."
pip install pytest black flake8

echo "Creating requirements.txt file..."
pip freeze > requirements.txt

echo "Environment setup complete!"
echo "Activate the environment with: source env/bin/activate"
