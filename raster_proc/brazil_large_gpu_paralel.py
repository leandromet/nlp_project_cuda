import rasterio
from rasterio.windows import Window
import numpy as np
import zarr
import os
import logging
from tqdm import tqdm
import multiprocessing
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import tempfile
import psutil

# Configuration
MAX_MEMORY_GB = 70
TILE_SIZE = 2048  # Smaller tile size for memory control
MAX_WORKERS = 10
MEMORY_BUFFER_GB = 5  # Leave some memory headroom

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grid_analysis_memopt.log'),
        logging.StreamHandler()
    ]
)

def get_available_memory():
    """Get available memory in GB with buffer"""
    return (psutil.virtual_memory().available / (1024 ** 3)) - MEMORY_BUFFER_GB

def calculate_changes_tile(args):
    """
    Calculate changes for a single tile across all years.
    Uses memory-mapped files to avoid loading all data at once.
    """
    tile_idx, vrt_path, window, temp_dir = args
    y, x = tile_idx
    
    # Create memory-mapped files for this tile's data
    tile_shape = (window.height, window.width)
    mmap_file = os.path.join(temp_dir, f'tile_{y}_{x}.dat')
    
    # Arrays to store results
    changes = np.zeros(tile_shape, dtype='uint8')
    transition_matrix = np.zeros((256, 256), dtype='uint64')
    
    with rasterio.open(vrt_path) as src:
        # Read first year to initialize
        prev_data = src.read(1, window=window)
        
        # Process subsequent years
        for year_idx in tqdm(range(2, src.count + 1), desc=f"Tile {y},{x}", leave=False):
            current_data = src.read(year_idx, window=window)
            
            # Calculate changes
            changed = prev_data != current_data
            changes += changed
            
            # Update transition matrix
            if np.any(changed):
                from_vals = prev_data[changed]
                to_vals = current_data[changed]
                
                for from_val, to_val in zip(from_vals, to_vals):
                    transition_matrix[from_val, to_val] += 1
            
            prev_data = current_data
    
    return changes, transition_matrix, (y, x)

def extract_grid_data_memopt(vrt_path, polygon_coords, output_dir):
    """Memory-optimized version for large 10x10 degree windows"""
    # Define or import the get_grid_name function
    def get_grid_name(polygon_coords):
        """Generate a grid name based on polygon coordinates."""
        return f"grid_{int(polygon_coords[0][0][0])}_{int(polygon_coords[0][0][1])}"

    grid_name = get_grid_name(polygon_coords)
    grid_output_dir = os.path.join(output_dir, grid_name)
    os.makedirs(grid_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(vrt_path) as src:
        # Calculate window (same as before)
        transform = src.transform
        lons = [coord[0] for coord in polygon_coords[0]]
        lats = [coord[1] for coord in polygon_coords[0]]
        lon_range = (min(lons), max(lons))
        lat_range = (min(lats), max(lats))
        
        ul_col = int((lon_range[0] - transform.c) / transform.a)
        ul_row = int((lat_range[1] - transform.f) / transform.e)
        lr_col = int((lon_range[1] - transform.c) / transform.a)
        lr_row = int((lat_range[0] - transform.f) / transform.e)
        
        ul_row = max(0, ul_row)
        ul_col = max(0, ul_col)
        lr_row = min(src.height, lr_row)
        lr_col = min(src.width, lr_col)
        
        full_window = Window.from_slices(
            rows=(ul_row, lr_row),
            cols=(ul_col, lr_col)
        )
        
        logging.info(f"Processing {grid_name} with window size: {full_window.height}x{full_window.width}")
        
        # Create temporary directory for memory-mapped files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Calculate how many tiles we need
            num_tiles_y = (full_window.height + TILE_SIZE - 1) // TILE_SIZE
            num_tiles_x = (full_window.width + TILE_SIZE - 1) // TILE_SIZE
            
            # Prepare arguments for parallel processing
            tasks = []
            for y in range(num_tiles_y):
                for x in range(num_tiles_x):
                    # Calculate tile window
                    tile_window = Window(
                        col_off=full_window.col_off + x * TILE_SIZE,
                        row_off=full_window.row_off + y * TILE_SIZE,
                        width=min(TILE_SIZE, full_window.width - x * TILE_SIZE),
                        height=min(TILE_SIZE, full_window.height - y * TILE_SIZE)
                    )
                    tasks.append(((y, x), vrt_path, tile_window, temp_dir))
            
            # Initialize results arrays
            full_changes = np.zeros((full_window.height, full_window.width), dtype='uint8')
            full_transitions = np.zeros((256, 256), dtype='uint64')
            
            # Process tiles in parallel with memory control
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                running_mem = 0
                
                for task in tasks:
                    # Estimate memory needed for this task (1 tile + results)
                    tile_mem = (TILE_SIZE * TILE_SIZE * 4) / (1024 ** 3)  # ~1MB for 512x512
                    
                    # Wait if we're approaching memory limits
                    while running_mem + tile_mem > get_available_memory():
                        completed = [f for f in futures if f.done()]
                        for f in completed:
                            changes, transitions, (y, x) = f.result()
                            running_mem -= tile_mem
                            
                            # Store results
                            y_start = y * TILE_SIZE
                            x_start = x * TILE_SIZE
                            y_end = min(y_start + TILE_SIZE, full_window.height)
                            x_end = min(x_start + TILE_SIZE, full_window.width)
                            
                            full_changes[y_start:y_end, x_start:x_end] = changes
                            full_transitions += transitions
                        
                        futures = [f for f in futures if not f.done()]
                    
                    # Submit new task
                    futures.append(executor.submit(calculate_changes_tile, task))
                    running_mem += tile_mem
                
                # Process remaining futures
                for f in tqdm(as_completed(futures), total=len(futures), desc="Finalizing tiles"):
                    changes, transitions, (y, x) = f.result()
                    
                    y_start = y * TILE_SIZE
                    x_start = x * TILE_SIZE
                    y_end = min(y_start + TILE_SIZE, full_window.height)
                    x_end = min(x_start + TILE_SIZE, full_window.width)
                    
                    full_changes[y_start:y_end, x_start:x_end] = changes
                    full_transitions += transitions
            
            # Create Zarr store
            zarr_path = os.path.join(output_dir, 'data.zarr')
            root = zarr.open(zarr_path, mode='w')
            
            # Store changes and transitions
            root.zeros(name='changes', shape=full_changes.shape, chunks=(512, 512), dtype='uint8')[:] = full_changes
            root.zeros(name='transitions', shape=full_transitions.shape, dtype='uint64')[:] = full_transitions
            
            # Store metadata
            window_transform = src.window_transform(full_window)
            root.attrs.update({
                'window_transform': json.dumps({
                    'a': window_transform.a,
                    'b': window_transform.b,
                    'c': window_transform.c,
                    'd': window_transform.d,
                    'e': window_transform.e,
                    'f': window_transform.f
                }),
                'crs': str(src.crs),
                'height': full_window.height,
                'width': full_window.width,
                'bounds': src.window_bounds(full_window),
                'lat_range': lat_range,
                'lon_range': lon_range,
                'grid_name': grid_name
            })
            
            # Store the first and last year for visualization
            first_year = src.read(1, window=full_window)
            last_year = src.read(src.count, window=full_window)
            
            root.zeros('first_year', shape=first_year.shape, dtype=first_year.dtype)[:] = first_year
            root.zeros('last_year', shape=last_year.shape, dtype=last_year.dtype)[:] = last_year
            
            return zarr_path

if __name__ == '__main__':
    VRT_FILE = '/srv/extrassd/2025_mapbiomas/mapbiomas_coverage_1985_2023.vrt'
    OUTPUT_BASE_DIR = 'grid_results_memopt'
    
    # Example 10x10 degree polygon
    POLYGON_10x10 = [((-54, 0), (-44, 0), (-44, -10), (-54, -10), (-54, 0))]
    
    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        logging.info(f"Starting memory-optimized analysis")
        
        zarr_path = extract_grid_data_memopt(VRT_FILE, POLYGON_10x10, OUTPUT_BASE_DIR)
        
        logging.info(f"Analysis complete. Results saved to {OUTPUT_BASE_DIR}")
        
    except Exception as e:
        logging.critical(f"Analysis failed: {str(e)}", exc_info=True)