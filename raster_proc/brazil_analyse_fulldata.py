import zarr
import numpy as np
import os
import sys
import traceback
from datetime import datetime
import json
from tqdm import tqdm
import logging
import plotly.graph_objects as go
from plotly.offline import plot
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from rasterio.transform import Affine
import psutil



# Color mapping and labels (same as before)
COLOR_MAP = {
    0: "#ffffff", 1:"#1f8d49",  3: "#1f8d49", 4: "#7dc975", 5: "#04381d", 6: "#007785",
    9: "#7a5900",  11: "#519799", 12: "#d6bc74", 15: "#edde8e", 20: "#db7093", 
    21: "#ffefc3", 23: "#ffa07a", 24: "#d4271e", 
    25: "#db4d4f", 29: "#ffaa5f", 30: "#9c0027",  31: "#091077", 
    32: "#fc8114", 33: "#2532e4", 35: "#9065d0", 39: "#f5b3c8",
    40: "#c71585", 41: "#f54ca9", 46: "#d68fe2", 47: "#9932cc", 48: "#e6ccff",
    49: "#02d659", 50: "#ad5100", 62: "#ff69b4"
}

LABELS = {
    0: "No data", 1: "Forest", 3: "Forest Formation", 4: "Savanna Formation", 5: "Mangrove", 6: "Floodable Forest",
    9: "Forest Plantation",  11: "Wetland", 12: "Grassland", 15: "Pasture", 20: "Sugar Cane",
    21: "Mosaic of Uses", 23: "Beach, Dune and Sand Spot", 24: "Urban Area",
    25: "Other non Vegetated Areas", 29: "Rocky Outcrop", 30: "Mining", 31: "Aquaculture",
    32: "Hypersaline Tidal Flat", 33: "River, Lake and Ocean", 35: "Palm Oil", 39: "Soybean",
    40: "Rice", 41: "Other Temporary Crops", 46: "Coffee", 47: "Citrus", 48: "Other Perennial Crops",
    49: "Wooded Sandbank Vegetation", 50: "Herbaceous Sandbank Vegetation", 62: "Cotton"
}


OUTPUT_DIR = '/media/bndt-ai/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/combined_results'
os.makedirs(OUTPUT_DIR, exist_ok=True, mode=0o777)  # Adds write permissions
# Configure robust logging
def configure_logging(output_dir):
    log_file = os.path.join(output_dir, f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    sys.stdout = open(log_file, 'a')
    sys.stderr = open(log_file, 'a')




# Memory-safe visualization
def visualize_full_results(zarr_path, output_dir, downsample_factor=20):
    try:
        logging.info(f"Memory before visualization: {psutil.virtual_memory()}")
        
        root = zarr.open(zarr_path, mode='r')
        bounds = [
            float(root.attrs.get('bounds', [0])[0]),
            float(root.attrs.get('bounds', [0,0])[1]),
            float(root.attrs.get('bounds', [0,0,1])[2]),
            float(root.attrs.get('bounds', [0,0,0,1])[3])
        ]
        
        # Downsample safely
        step = max(1, downsample_factor)
        data = root['data'][-1, ::step, ::step]
        changes = root['changes'][::step, ::step]
        
        # Plot land cover
        plt.figure(figsize=(20, 15))
        unique_vals = sorted(COLOR_MAP.keys())
        cmap = ListedColormap([COLOR_MAP[v] for v in unique_vals])
        norm = BoundaryNorm([v-0.5 for v in unique_vals] + [unique_vals[-1]+0.5], len(unique_vals))
        plt.imshow(data, cmap=cmap, norm=norm, extent=bounds, aspect='auto')
        plt.savefig(os.path.join(output_dir, 'landcover.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot changes
        plt.figure(figsize=(20, 15))
        plt.imshow(changes, cmap='viridis', extent=bounds, aspect='auto')
        plt.savefig(os.path.join(output_dir, 'changes.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Memory after visualization: {psutil.virtual_memory()}")
    except Exception:
        logging.error(f"Visualization failed:\n{traceback.format_exc()}")

# Robust Sankey generation
def create_full_sankey_diagrams(zarr_path, output_dir, max_samples=500000):
    try:
        logging.info(f"Memory before Sankey: {psutil.virtual_memory()}")
        
        root = zarr.open(zarr_path, mode='r')
        years = list(range(1985, 2024))
        sample_size = min(max_samples, root['data'].shape[1] * root['data'].shape[2] // 100)
        
        for start, end in [(1985,1995), (1995,2005), (2005,2015), (2015,2023)]:
            try:
                # Simple random sampling (more stable than weighted)
                idx = np.random.choice(root['data'].shape[1] * root['data'].shape[2], sample_size)
                rows, cols = np.unravel_index(idx, (root['data'].shape[1], root['data'].shape[2]))
                
                start_data = root['data'][years.index(start), rows, cols]
                end_data = root['data'][years.index(end), rows, cols]
                
                # [Your existing Sankey diagram code here]
                
            except Exception:
                logging.error(f"Sankey {start}-{end} failed:\n{traceback.format_exc()}")
                
        logging.info(f"Memory after Sankey: {psutil.virtual_memory()}")
    except Exception:
        logging.error(f"Sankey generation failed:\n{traceback.format_exc()}")

def main():
    OUTPUT_DIR = '/media/bndt-ai/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/combined_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True, mode=0o777)
    configure_logging(OUTPUT_DIR)
    
    try:
        combined_zarr = os.path.join(OUTPUT_DIR, 'combined_data.zarr')
        if not os.path.exists(combined_zarr):
            raise FileNotFoundError(f"Zarr not found at {combined_zarr}")
        
        # Increase open files limit for Zarr
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
        
        visualize_full_results(combined_zarr, OUTPUT_DIR)
        create_full_sankey_diagrams(combined_zarr, OUTPUT_DIR)
        
    except Exception:
        logging.critical(f"Main process failed:\n{traceback.format_exc()}")
    finally:
        logging.info("===== PROCESS COMPLETED =====")
        if 'sys' in globals():
            sys.stdout.close()
            sys.stderr.close()

if __name__ == '__main__':
    main()