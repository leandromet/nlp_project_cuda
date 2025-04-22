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
matplotlib.use('Agg')  # Non-interactive backend
import plotly.graph_objects as go
from plotly.offline import plot
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
MAX_MEMORY_GB = 70
TILE_SIZE = 2048
MAX_WORKERS = 10
MEMORY_BUFFER_GB = 5

# Color mapping and labels (same as your original)
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
    """Calculate changes for a single tile across all years."""
    tile_idx, vrt_path, window, temp_dir = args
    y, x = tile_idx
    
    # Arrays to store results
    changes = np.zeros((window.height, window.width), dtype='uint8')
    transition_matrix = np.zeros((256, 256), dtype='uint64')
    
    from rasterio.env import Env
    
    with Env(GDAL_TIFF_INTERNAL_MASK=False, GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(vrt_path) as src:
            # Read first year to initialize
            prev_data = src.read(1, window=window)
        
        # Process subsequent years
        for year_idx in range(2, src.count + 1):
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

def get_grid_name(polygon_coords):
    """Generate a grid name from polygon coordinates."""
    lons = [coord[0] for coord in polygon_coords[0]]
    lats = [coord[1] for coord in polygon_coords[0]]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    return f"grid_{int(abs(max_lat))}_{int(abs(min_lon))}"

def extract_grid_data_memopt(vrt_path, polygon_coords, output_dir):
    """Memory-optimized extraction with visualization."""
    grid_name = get_grid_name(polygon_coords)
    grid_output_dir = os.path.join(output_dir, grid_name)
    os.makedirs(grid_output_dir, exist_ok=True)
    
    with rasterio.open(vrt_path) as src:
        # Calculate window
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
        
        full_window = Window.from_slices(rows=(ul_row, lr_row), cols=(ul_col, lr_col))
        
        logging.info(f"Processing {grid_name} with window size: {full_window.height}x{full_window.width}")
        
        # Create preview image (downsampled)
        preview_window = Window(
            col_off=full_window.col_off,
            row_off=full_window.row_off,
            width=min(1000, full_window.width),
            height=min(1000, full_window.height)
        )
        preview_data = src.read(1, window=preview_window, out_shape=(1000, 1000))
        
        plt.figure(figsize=(10, 10))
        plt.imshow(preview_data, cmap='viridis')
        plt.title(f"Preview of {grid_name}")
        plt.colorbar(label='Land Cover Class')
        plt.savefig(os.path.join(grid_output_dir, 'extraction_preview.png'))
        plt.close()
        
        # Process data in tiles
        with tempfile.TemporaryDirectory() as temp_dir:
            num_tiles_y = (full_window.height + TILE_SIZE - 1) // TILE_SIZE
            num_tiles_x = (full_window.width + TILE_SIZE - 1) // TILE_SIZE
            
            tasks = [((y, x), vrt_path, 
                     Window(
                         col_off=full_window.col_off + x * TILE_SIZE,
                         row_off=full_window.row_off + y * TILE_SIZE,
                         width=min(TILE_SIZE, full_window.width - x * TILE_SIZE),
                         height=min(TILE_SIZE, full_window.height - y * TILE_SIZE)
                     ), temp_dir)
                    for y in range(num_tiles_y) for x in range(num_tiles_x)]
            
            full_changes = np.zeros((full_window.height, full_window.width), dtype='uint8')
            full_transitions = np.zeros((256, 256), dtype='uint64')
            
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                running_mem = 0
                
                for task in tasks:
                    tile_mem = (TILE_SIZE * TILE_SIZE * 4) / (1024 ** 3)
                    
                    while running_mem + tile_mem > get_available_memory():
                        completed = [f for f in futures if f.done()]
                        for f in completed:
                            changes, transitions, (y, x) = f.result()
                            running_mem -= tile_mem
                            
                            y_start = y * TILE_SIZE
                            x_start = x * TILE_SIZE
                            y_end = min(y_start + TILE_SIZE, full_window.height)
                            x_end = min(x_start + TILE_SIZE, full_window.width)
                            
                            full_changes[y_start:y_end, x_start:x_end] = changes
                            full_transitions += transitions
                        
                        futures = [f for f in futures if not f.done()]
                    
                    futures.append(executor.submit(calculate_changes_tile, task))
                    running_mem += tile_mem
                
                for f in tqdm(as_completed(futures), total=len(futures), desc="Processing tiles"):
                    changes, transitions, (y, x) = f.result()
                    y_start = y * TILE_SIZE
                    x_start = x * TILE_SIZE
                    y_end = min(y_start + TILE_SIZE, full_window.height)
                    x_end = min(x_start + TILE_SIZE, full_window.width)
                    
                    full_changes[y_start:y_end, x_start:x_end] = changes
                    full_transitions += transitions
            
            # Create Zarr store
            zarr_path = os.path.join(grid_output_dir, 'data.zarr')
            root = zarr.open(zarr_path, mode='w')
            
            # Store data
            root.zeros('changes', shape=full_changes.shape, chunks=(512, 512), dtype='uint8')[:] = full_changes
            root.zeros('transitions', shape=full_transitions.shape, dtype='uint64')[:] = full_transitions
            
            # Store first and last year
            first_year = src.read(1, window=full_window)
            last_year = src.read(src.count, window=full_window)
            root.zeros('first_year', shape=first_year.shape, dtype=first_year.dtype)[:] = first_year
            root.zeros('last_year', shape=last_year.shape, dtype=last_year.dtype)[:] = last_year
            
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
                'grid_name': grid_name
            })
            
            return zarr_path, grid_output_dir

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
    cmap = ListedColormap([COLOR_MAP.get(i, '#ffffff') for i in range(max(COLOR_MAP.keys())+1)])
    norm = matplotlib.colors.BoundaryNorm(list(COLOR_MAP.keys()), len(COLOR_MAP))
    
    plt.figure(figsize=(12, 10))
    plt.imshow(last_year, cmap=cmap, norm=norm)
    plt.title(f"{grid_name} Land Cover")
    
    # Create legend
    patches = [plt.Patch(color=COLOR_MAP[k], label=f"{k}: {LABELS[k]}") 
              for k in sorted(LABELS.keys()) if k in COLOR_MAP]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'landcover_map.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_changes_map(root, output_dir):
    """Create changes frequency visualization."""
    grid_name = root.attrs['grid_name']
    changes = root['changes'][:]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(changes, cmap='viridis')
    plt.colorbar(label='Number of Changes (1985-2023)')
    plt.title(f"{grid_name} Change Frequency")
    plt.savefig(os.path.join(output_dir, 'changes_map.png'), dpi=300)
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
    link_colors = [f"{COLOR_MAP.get(s, '#999999')}88" for s in sources]
    
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
    
    if not os.path.exists(VRT_FILE):
        logging.critical(f"VRT file not found: {VRT_FILE}")
        raise FileNotFoundError(f"VRT file not found: {VRT_FILE}")
    OUTPUT_BASE_DIR = 'grid_results_memopt'
    
    # Example 10x10 degree polygon
    POLYGON_10x10 = [((-54, 0), (-44, 0), (-44, -10), (-54, -10), (-54, 0))]
    
    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        logging.info("Starting memory-optimized analysis with visualization")
        
        zarr_path, grid_output_dir = extract_grid_data_memopt(VRT_FILE, POLYGON_10x10, OUTPUT_BASE_DIR)
        create_visualizations(zarr_path, grid_output_dir)
        
        logging.info(f"Analysis complete. Results saved to {grid_output_dir}")
        
    except Exception as e:
        logging.critical(f"Analysis failed: {str(e)}", exc_info=True)