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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
from plotly.offline import plot
import warnings
import time
from functools import partial

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
MAX_MEMORY_GB = 70

def get_available_memory():
    """Calculate available memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)
TILE_SIZE = 256
MAX_WORKERS = 20
MEMORY_BUFFER_GB = 8
MAX_READ_RETRIES = 3
RETRY_DELAY = 0.05  # seconds

# Color mapping and labels (same as before)
COLOR_MAP = {
    0: "#ffffff", 1: "#1f8d49", 3: "#1f8d49", 4: "#7dc975", 5: "#04381d", 6: "#007785",
    9: "#7a5900", 11: "#519799", 12: "#d6bc74", 15: "#edde8e", 20: "#db7093",
    21: "#ffefc3", 23: "#ffa07a", 24: "#d4271e", 25: "#db4d4f", 29: "#ffaa5f",
    30: "#9c0027", 31: "#091077", 32: "#fc8114", 33: "#2532e4", 35: "#9065d0",
    39: "#f5b3c8", 40: "#c71585", 41: "#f54ca9", 46: "#d68fe2", 47: "#9932cc",
    48: "#e6ccff", 49: "#02d659", 50: "#ad5100", 62: "#ff69b4"
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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grid_analysis_robust.log'),
        logging.StreamHandler()
    ]
)

def robust_read(src, band, window, retries=MAX_READ_RETRIES):
    """Attempt to read a tile with retries on failure."""
    for attempt in range(retries):
        try:
            # Clip the window to the valid bounds of the raster
            valid_window = window.intersection(Window(0, 0, src.width, src.height))
            if valid_window is None or valid_window.width <= 0 or valid_window.height <= 0:
                raise ValueError("Requested window is outside the raster bounds")
            data = src.read(band, window=valid_window)
            # Basic data validation
            if np.all(data == src.nodata):
                raise ValueError("All values are nodata")
            return data
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(RETRY_DELAY * (attempt + 1))
            continue
    raise RuntimeError(f"Failed after {retries} attempts")

def process_tile(task):
                    """Process a single tile for changes and transitions."""
                    y, x, path, window, temp_dir = task
                    try:
                        with rasterio.open(path) as src:
                            changes = np.zeros((window.height, window.width), dtype='uint8')
                            transitions = np.zeros((256, 256), dtype='uint64')
                            
                            # Read first band
                            FILL_VALUE = 0  # Define a default fill value
                            prev_data = src.read(1, window=window, fill_value=FILL_VALUE)
                            
                            # Process subsequent bands
                            for band_idx in range(2, src.count + 1):
                                curr_data = src.read(band_idx, window=window, fill_value=FILL_VALUE)
                                changed = prev_data != curr_data
                                changes += changed
                                
                                # Update transition matrix
                                if np.any(changed):
                                    from_vals = prev_data[changed]
                                    to_vals = curr_data[changed]
                                    for f, t in zip(from_vals, to_vals):
                                        transitions[f, t] += 1
                                
                                prev_data = curr_data
                            
                            return changes, transitions, (y, x)
                    except Exception as e:
                        logging.warning(f"Tile {y},{x} failed: {str(e)}")
                        return None
                    

def calculate_changes_tile(args):
    """Calculate changes for a single tile with robust error handling."""
    tile_idx, vrt_path, window, temp_dir = args
    y, x = tile_idx
    
    changes = np.zeros((window.height, window.width), dtype='uint8')
    transition_matrix = np.zeros((256, 256), dtype='uint64')
    failed = False
    
    try:
        with rasterio.open(vrt_path) as src:
            # Read first year with retries
            try:
                prev_data = robust_read(src, 1, window)
            except Exception as e:
                logging.warning(f"Failed reading tile {y},{x} (band 1): {str(e)}")
                failed = True
                return None
            
            # Process subsequent years
            for year_idx in range(2, src.count + 1):
                try:
                    current_data = robust_read(src, year_idx, window)
                except Exception as e:
                    logging.warning(f"Failed reading tile {y},{x} (band {year_idx}): {str(e)}")
                    failed = True
                    break
                
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
            
            if not failed:
                return changes, transition_matrix, (y, x)
            return None
            
    except Exception as e:
        logging.error(f"Critical error processing tile {y},{x}: {str(e)}")
        return None

def get_grid_name(polygon_coords):
    """Generate a grid name based on polygon coordinates."""
    min_lon = min(coord[0] for coord in polygon_coords[0])
    max_lon = max(coord[0] for coord in polygon_coords[0])
    min_lat = min(coord[1] for coord in polygon_coords[0])
    max_lat = max(coord[1] for coord in polygon_coords[0])
    return f"grid_{min_lon}_{max_lon}_{min_lat}_{max_lat}"




def extract_grid_data_optimized(vrt_path, polygon_coords, output_dir):
    """Optimized extraction with VRT-aware settings."""
    # VRT-specific configuration
    VRT_BLOCK_SIZE = 512  # Matches the BlockXSize/BlockYSize in VRT
    TILE_SIZE = 2048      # Should be a multiple of VRT_BLOCK_SIZE
    MAX_WORKERS = 8       # Conservative worker count
    FILL_VALUE = 0        # Matches NoDataValue in VRT
    
    def get_grid_name(polygon_coords):
        """Generate grid name from coordinates."""
        coords = polygon_coords[0]
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)
        return f"grid_{abs(min_lon):.0f}W_{abs(max_lon):.0f}W_{abs(min_lat):.0f}S_{abs(max_lat):.0f}S"

    grid_name = get_grid_name(polygon_coords)
    grid_output_dir = os.path.join(output_dir, grid_name)
    os.makedirs(grid_output_dir, exist_ok=True)

    # Special rasterio configuration for problematic TIFFs
    rasterio_config = {
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',  # Prevent scanning of all files
        'GDAL_TIFF_OVR_BLOCKSIZE': '512',             # Match internal block size
        'CHECK_DISK_FREE_SPACE': 'NO',
        'GDAL_MAX_DATASET_POOL_SIZE': '200'           # Increase dataset pool
    }

    try:
        with rasterio.Env(**rasterio_config), rasterio.open(vrt_path) as src:
            # Verify we're reading the VRT correctly
            logging.info(f"Dataset info - Width: {src.width}, Height: {src.height}, Bands: {src.count}")
            logging.info(f"Block sizes: {src.block_shapes}")
            logging.info(f"Driver: {src.driver}, CRS: {src.crs}")
            
            # Calculate window (aligned to VRT block size)
            transform = src.transform
            coords = polygon_coords[0]
            lons = [coord[0] for coord in coords]
            lats = [coord[1] for coord in coords]
            lon_range = (min(lons), max(lons))
            lat_range = (min(lats), max(lats))
            
            # Convert coordinates to pixel locations (aligned to blocks)
            def align_to_blocks(value, block_size, max_value, align_type='floor'):
                if align_type == 'floor':
                    return max(0, (value // block_size) * block_size)
                else:
                    return min(max_value, ((value + block_size - 1) // block_size) * block_size)
            
            ul_col = align_to_blocks(int((lon_range[0] - transform.c) / transform.a), VRT_BLOCK_SIZE, src.width)
            ul_row = align_to_blocks(int((lat_range[1] - transform.f) / transform.e), VRT_BLOCK_SIZE, src.height)
            lr_col = align_to_blocks(int((lon_range[1] - transform.c) / transform.a), VRT_BLOCK_SIZE, src.width, 'ceil')
            lr_row = align_to_blocks(int((lat_range[0] - transform.f) / transform.e), VRT_BLOCK_SIZE, src.height, 'ceil')
            
            full_window = Window.from_slices(
                rows=(ul_row, lr_row),
                cols=(ul_col, lr_col)
            )
            
            logging.info(f"Processing {grid_name} with aligned window size: {full_window.height}x{full_window.width}")
            
            # Create preview using overviews if available
            try:
                with rasterio.open(vrt_path, OVERVIEW_LEVEL=2) as ovr_src:
                    ovr_window = Window(
                        col_off=full_window.col_off / 4,
                        row_off=full_window.row_off / 4,
                        width=full_window.width / 4,
                        height=full_window.height / 4
                    )
                    preview_data = ovr_src.read(1, window=ovr_window)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(preview_data, cmap='viridis')
                    plt.title(f"Preview of {grid_name} (from overview)")
                    plt.colorbar(label='Land Cover Class')
                    plt.savefig(os.path.join(grid_output_dir, 'extraction_preview.png'))
                    plt.close()
            except Exception as e:
                logging.warning(f"Couldn't use overviews for preview: {str(e)}")
                try:
                    preview_data = src.read(1, window=full_window, out_shape=(1000, 1000))
                    plt.figure(figsize=(10, 10))
                    plt.imshow(preview_data, cmap='viridis')
                    plt.title(f"Preview of {grid_name}")
                    plt.colorbar(label='Land Cover Class')
                    plt.savefig(os.path.join(grid_output_dir, 'extraction_preview.png'))
                    plt.close()
                except Exception as e:
                    logging.error(f"Failed to create preview: {str(e)}")
            
            # Process data in blocks aligned with VRT structure
            with tempfile.TemporaryDirectory() as temp_dir:
                num_tiles_y = (full_window.height + TILE_SIZE - 1) // TILE_SIZE
                num_tiles_x = (full_window.width + TILE_SIZE - 1) // TILE_SIZE
                
                tasks = []
                for y in range(num_tiles_y):
                    for x in range(num_tiles_x):
                        tile_window = Window(
                            col_off=full_window.col_off + x * TILE_SIZE,
                            row_off=full_window.row_off + y * TILE_SIZE,
                            width=min(TILE_SIZE, full_window.width - x * TILE_SIZE),
                            height=min(TILE_SIZE, full_window.height - y * TILE_SIZE)
                        )
                        tasks.append((y, x, vrt_path, tile_window, temp_dir))
                
                full_changes = np.zeros((full_window.height, full_window.width), dtype='uint8')
                full_transitions = np.zeros((256, 256), dtype='uint64')
                failed_tiles = []
                
                
                
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = []
                    for task in tasks:
                        futures.append(executor.submit(process_tile, task))
                    
                    progress = tqdm(as_completed(futures), total=len(futures), desc="Processing tiles")
                    for future in progress:
                        result = future.result()
                        if result is not None:
                            changes, transitions, (y, x) = result
                            y_start = y * TILE_SIZE
                            x_start = x * TILE_SIZE
                            y_end = min(y_start + TILE_SIZE, full_window.height)
                            x_end = min(x_start + TILE_SIZE, full_window.width)
                            
                            full_changes[y_start:y_end, x_start:x_end] = changes
                            full_transitions += transitions
                        else:
                            failed_tiles.append((y, x))
                        progress.set_postfix(failed=len(failed_tiles))
                
                if failed_tiles:
                    logging.warning(f"{len(failed_tiles)} tiles failed to process")
                    np.save(os.path.join(grid_output_dir, 'failed_tiles.npy'), np.array(failed_tiles))
                
                # Create Zarr store with compression
                zarr_path = os.path.join(grid_output_dir, 'data.zarr')
                root = zarr.open(zarr_path, mode='w')
                
                # Store data
                root.zeros(
                    'changes',
                    shape=full_changes.shape,
                    chunks=(VRT_BLOCK_SIZE, VRT_BLOCK_SIZE)
                )[:] = full_changes
                
                root.zeros(
                    'transitions',
                    shape=full_transitions.shape,
                    dtype='uint64'
                )[:] = full_transitions
                
                # Store first and last year
                try:
                    first_year = src.read(1, window=full_window, fill_value=FILL_VALUE)
                    root.zeros(
                        'first_year',
                        shape=first_year.shape,
                        dtype=first_year.dtype,
                        chunks=(VRT_BLOCK_SIZE, VRT_BLOCK_SIZE)
                    )[:] = first_year
                except Exception as e:
                    logging.error(f"Failed to store first year: {str(e)}")
                
                try:
                    last_year = src.read(src.count, window=full_window, fill_value=FILL_VALUE)
                    root.zeros(
                        'last_year',
                        shape=last_year.shape,
                        dtype=last_year.dtype,
                        chunks=(VRT_BLOCK_SIZE, VRT_BLOCK_SIZE)
                    )[:] = last_year
                except Exception as e:
                    logging.error(f"Failed to store last year: {str(e)}")
                
                # Store metadata
                window_transform = src.window_transform(full_window)
                root.attrs.update({
                    'window_transform': json.dumps(window_transform.to_gdal()),
                    'crs': str(src.crs),
                    'height': full_window.height,
                    'width': full_window.width,
                    'bounds': src.window_bounds(full_window),
                    'lat_range': lat_range,
                    'lon_range': lon_range,
                    'grid_name': grid_name,
                    'failed_tiles': len(failed_tiles),
                    'processing_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'vrt_block_size': VRT_BLOCK_SIZE,
                    'fill_value': FILL_VALUE
                })
                
                return zarr_path, grid_output_dir

    except Exception as e:
        logging.critical(f"Error processing grid {grid_name}: {str(e)}")
        raise



def create_visualizations(zarr_path, output_dir):
    """Create all visualizations from the Zarr data."""
    root = zarr.open(zarr_path, mode='r')
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    # 1. Create land cover map
    create_landcover_map(root, output_dir)
    
    # 2. Create changes map
    create_changes_map(root, output_dir)
    
    # 3. Create Sankey diagrams
    create_sankey_diagrams(root, output_dir)

def create_landcover_map(root, output_dir):
    """Create land cover visualization."""
    grid_name = root.attrs['grid_name']
    last_year = root['last_year'][:]
    
    # Create colormap
    cmap = ListedColormap([COLOR_MAP.get(i, '#ffffff') for i in range(max(COLOR_MAP.keys()) + 1)])
    
    plt.figure(figsize=(12, 10))
    plt.imshow(last_year, cmap=cmap, vmin=0, vmax=max(COLOR_MAP.keys()))
    plt.title(f"{grid_name} Land Cover")
    
    # Create legend
    from matplotlib.patches import Patch
    patches = [Patch(color=COLOR_MAP[k], label=f"{k}: {LABELS[k]}") 
               for k in sorted(LABELS.keys()) if k in COLOR_MAP]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'landcover_map.png'), dpi=450, bbox_inches='tight')
    plt.close()

def create_changes_map(root, output_dir):
    """Create changes frequency visualization."""
    grid_name = root.attrs['grid_name']
    changes = root['changes'][:]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(changes, cmap='gist_stern')
    plt.colorbar(label='Number of Changes (1985-2023)')
    plt.title(f"{grid_name} Change Frequency")
    plt.savefig(os.path.join(output_dir, 'changes_map.png'), dpi=450)
    plt.close()

def create_sankey_diagrams(root, output_dir):
    """Create decadal Sankey diagrams."""
    grid_name = root.attrs['grid_name']
    transitions = root['transitions'][:]
    first_year = root['first_year'][:]
    last_year = root['last_year'][:]
    
    # Create full-period Sankey
    create_sankey(transitions, f"{grid_name} (1985-2023)", 
                 os.path.join(output_dir, 'transitions_full.html'))
    
    # You could add decadal breakdowns here if you stored yearly data

def create_sankey(transition_matrix, title, output_path):
    """Create a single Sankey diagram."""
    # Filter small transitions
    threshold = np.sum(transition_matrix) * 0.001
    significant = transition_matrix > threshold
    
    # Prepare nodes and links
    sources, targets, values = [], [], []
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            if significant[i, j]:
                sources.append(i)
                targets.append(j)
                values.append(transition_matrix[i, j])
    
    # Create node labels with class information
    node_labels = [f"{i}: {LABELS.get(i, 'Unknown')}" for i in set(sources).union(set(targets))]
    
    # Create colors
    node_colors = [COLOR_MAP.get(i, '#999999') for i in set(sources).union(set(targets))]
    link_colors = [COLOR_MAP.get(s, '#999999') for s in sources]
    
    # Create figure
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    ))
    
    fig.update_layout(title_text=title, font_size=10)
    plot(fig, filename=output_path, auto_open=False)

if __name__ == '__main__':
    VRT_FILE = '/srv/extrassd/2025_mapbiomas/mapbiomas_coverage_1985_2023.vrt'
    OUTPUT_BASE_DIR = 'grid_results_robust'
    
    # Example 10x10 degree polygon
    POLYGON_8x8 = [((-54, 0), (-46, 0), (-46, -8), (-54, -8), (-54, 0))]
    
    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        logging.info("Starting robust analysis with visualization")
        
        zarr_path, grid_output_dir = extract_grid_data_optimized(VRT_FILE, POLYGON_8x8, OUTPUT_BASE_DIR)
        create_visualizations(zarr_path, grid_output_dir)
        
        logging.info(f"Analysis complete. Results saved to {grid_output_dir}")
        
    except Exception as e:
        logging.critical(f"Analysis failed: {str(e)}", exc_info=True)