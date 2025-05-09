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
from collections import defaultdict
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
import subprocess
import kaleido



# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
#MAX_MEMORY_GB = 70

def get_available_memory():
    """Calculate available memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)
TILE_SIZE = 256
MAX_WORKERS = 10
MEMORY_BUFFER_GB = 5
MAX_READ_RETRIES = 3
RETRY_DELAY = 0.05  # seconds

# Color mapping and labels (same as before)
COLOR_MAP = {
    0: "#ffffff",
    1: "#1f8d49", 3: "#1f8d49", 4: "#7dc975", 5: "#04381d", 6: "#007785",
    9: "#7a5900", 10: "#d6bc74", 11: "#519799", 12: "#d6bc74", 13:"#ffffff", 14: "#ffefc3",
    15:"#edde8e", 18: "#e974ed", 19:"#c27ba0", 20: "#db7093",   
    21: "#ffefc3", 22:"#d4271e", 23: "#ffa07a", 24: "#d4271e", 25: "#db4d4f", 26:"#2532e4", 29: "#ffaa5f",
    30: "#9c0027", 31: "#091077", 32: "#fc8114", 33: "#259fe4", 35: "#9065d0", 36:"#d082de",
    39: "#f5b3c8", 40: "#c71585", 41: "#f54ca9", 46: "#d68fe2", 47: "#9932cc",
    48: "#e6ccff", 49: "#02d659", 50: "#ad5100", 62: "#ff69b4", 27: "#ffffff"
}

LABELS = {
    0: "No data", 
    1: "Forest", 3: "Forest Formation", 4: "Savanna Formation", 5: "Mangrove", 6: "Floodable Forest",
    9: "Forest Plantation",  11: "Wetland", 10: "Herbaceous", 12: "Grassland",13:"other", 14:"Farming",
    15: "Pasture", 18:"Agri", 19:"Temporary Crop", 20: "Sugar Cane",
    21: "Mosaic of Uses", 22:"Non vegetated", 23: "Beach and Sand", 24: "Urban Area",
    25: "Other non Vegetated Areas", 26:"Water", 29: "Rocky Outcrop", 30: "Mining", 31: "Aquaculture",
    32: "Hypersaline Tidal Flat", 33: "River Lake and Ocean", 35: "Palm Oil", 36:"Perennial Crop", 39: "Soybean",
    40: "Rice", 41: "Other Temporary Crops", 46: "Coffee", 47: "Citrus", 48: "Other Perennial Crops",
    49: "Wooded Sandbank Vegetation", 50: "Herbaceous Sandbank Vegetation", 62: "Cotton", 27: "Not Observed"
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
                            FILL_VALUE = src.nodata if src.nodata is not None else 0  # Use raster's nodata value or default to 0
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
                    
                    # Ensure transitions are correctly aggregated without duplication
                    unique_transitions, counts = np.unique(
                        np.stack((from_vals, to_vals), axis=1), axis=0, return_counts=True
                    )
                    for (from_val, to_val), count in zip(unique_transitions, counts):
                        transition_matrix[from_val, to_val] += count
                
                prev_data = current_data
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
            # try:
            #     with rasterio.open(vrt_path, OVERVIEW_LEVEL=2) as ovr_src:
            #         ovr_window = Window(
            #             col_off=full_window.col_off / 4,
            #             row_off=full_window.row_off / 4,
            #             width=full_window.width / 4,
            #             height=full_window.height / 4
            #         )
            #         preview_data = ovr_src.read(1, window=ovr_window)
            #         plt.figure(figsize=(12, 20))
            #         plt.imshow(preview_data, cmap='viridis')
            #         plt.title(f"Preview of {grid_name} (from overview)")
            #         plt.colorbar(label='Land Cover Class')
            #         plt.savefig(os.path.join(grid_output_dir, 'extraction_preview.png'))
            #         plt.close()
            # except Exception as e:
            #     logging.warning(f"Couldn't use overviews for preview: {str(e)}")
            #     try:
            #         preview_data = src.read(1, window=full_window, out_shape=(1500, 1000))
            #         plt.figure(figsize=(12, 20))
            #         plt.imshow(preview_data, cmap='viridis')
            #         plt.title(f"Preview of {grid_name}")
            #         plt.colorbar(label='Land Cover Class')
            #         plt.savefig(os.path.join(grid_output_dir, 'extraction_preview.png'))
            #         plt.close()
            #     except Exception as e:
            #         logging.error(f"Failed to create preview: {str(e)}")
            
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

                # Store data with proper chunkingfor diagrams
                transform = src.transform
            
                # Convert geographic coordinates to pixel coordinates
                ul_col = int((lon_range[0] - transform.c) / transform.a)
                ul_row = int((lat_range[1] - transform.f) / transform.e)
                lr_col = int((lon_range[1] - transform.c) / transform.a)
                lr_row = int((lat_range[0] - transform.f) / transform.e)
                
                # Ensure window is within bounds
                ul_row = max(0, ul_row)
                ul_col = max(0, ul_col)
                lr_row = min(src.height, lr_row)
                lr_col = min(src.width, lr_col)
                
                window = Window.from_slices(
                    rows=(ul_row, lr_row),
                    cols=(ul_col, lr_col)
                )
                data = src.read(window=window)
                data_array = root.zeros(
                    'data',
                    shape=data.shape,
                    chunks=(1, 512, 512),
                    dtype=data.dtype
                )
                data_array[:] = data
                
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

    # 4. Create decadal Sankey diagrams
    create_decadal_sankey_diagrams(root, output_dir)

def create_landcover_map(root, output_dir):
    """Create land cover visualization with PNGW for GIS compatibility."""
    grid_name = root.attrs['grid_name']
    last_year = root['last_year'][:]
    transform = rasterio.transform.Affine(*json.loads(root.attrs['window_transform']))
    
    # Create colormap
    cmap = ListedColormap([COLOR_MAP.get(i, '#ffffff') for i in range(max(COLOR_MAP.keys()) + 1)])
    
    plt.figure(figsize=(12, 20))
    plt.imshow(last_year, cmap=cmap, vmin=0, vmax=max(COLOR_MAP.keys()))
    plt.title(f"{grid_name} Land Cover")
    
    # Create legend
    from matplotlib.patches import Patch
    patches = [Patch(color=COLOR_MAP[k], label=f"{k}: {LABELS[k]}") 
               for k in sorted(LABELS.keys()) if k in COLOR_MAP]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'landcover_map.png')
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()
    
    # Create PNGW file for georeferencing
    pngw_path = output_path + 'w'
    with open(pngw_path, 'w') as pngw_file:
        pngw_file.write(f"{transform.a}\n")  # Pixel size in x-direction
        pngw_file.write(f"{transform.b}\n")  # Rotation (usually 0)
        pngw_file.write(f"{transform.d}\n")  # Rotation (usually 0)
        pngw_file.write(f"{transform.e}\n")  # Pixel size in y-direction (negative)
        pngw_file.write(f"{transform.c}\n")  # X-coordinate of the upper-left corner
        pngw_file.write(f"{transform.f}\n")  # Y-coordinate of the upper-left corner

def create_changes_map(root, output_dir):
    """Create changes frequency visualization with PNGW for GIS compatibility."""
    grid_name = root.attrs['grid_name']
    changes = root['changes'][:]
    transform = json.loads(root.attrs['window_transform'])
    
    # Create the changes map
    plt.figure(figsize=(12, 20))
    plt.imshow(changes, cmap='gist_stern')
    plt.colorbar(label='Number of Changes (1985-2023)')
    plt.title(f"{grid_name} Change Frequency")
    
    # Save the PNG file
    output_path = os.path.join(output_dir, 'changes_map.png')
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()
    
    # Create PNGW file for georeferencing
    pngw_path = output_path + 'w'
    with open(pngw_path, 'w') as pngw_file:
        pngw_file.write(f"{transform[0]}\n")  # Pixel size in x-direction
        pngw_file.write(f"{transform[1]}\n")  # Rotation (usually 0)
        pngw_file.write(f"{transform[2]}\n")  # Rotation (usually 0)
        pngw_file.write(f"{transform[3]}\n")  # Pixel size in y-direction (negative)
        pngw_file.write(f"{transform[4]}\n")  # X-coordinate of the upper-left corner
        pngw_file.write(f"{transform[5]}\n")  # Y-coordinate of the upper-left corner

def create_sankey_diagrams(root, output_dir):
    """Create organized Sankey diagrams with proper class alignment."""
    grid_name = root.attrs['grid_name']
    
    # Get data as numpy arrays
    if 'data' in root:
        yearly_data = [np.asarray(root['data'][i][:]) for i in range(root['data'].shape[0])]
        years = list(range(1985, 1985 + len(yearly_data)))
        if len(yearly_data) != len(years):
            logging.error("Mismatch between yearly data and years. Adjusting to match available data.")
            years = years[:len(yearly_data)]
    else:
        yearly_data = [np.asarray(root['first_year'][:]), np.asarray(root['last_year'][:])]
        years = [1985, 2023]
    
    # Define decades
    decades = [
        (1985, 1995),  # 1985-1994 (10 years)
        (1995, 2005),  # 1995-2004 (10 years)
        (2005, 2015),  # 2005-2014 (10 years)
        (2015, 2023),   # 2015-2023 (9 years)
        (1985, 2023)  # Full period (38 years)
    ]
    
    # Get all present classes (only those with labels)
    present_classes = list(set(cls for data in yearly_data 
                            for cls in np.unique(data) 
                            if cls in LABELS))
    
    # Order classes by frequency (largest on top)
    class_freq = defaultdict(int)
    for data in yearly_data:
        unique, counts = np.unique(data, return_counts=True)
        for cls, cnt in zip(unique, counts):
            if cls in LABELS:
                class_freq[cls] += cnt
    present_classes = sorted(present_classes, key=lambda x: -class_freq[x])
    
    # Create decade diagrams
    for start_year, end_year in decades:
        decade_indices = [i for i, y in enumerate(years) if start_year <= y <= end_year]
        if len(decade_indices) < 2:
            continue  # Skip if not enough data
        
        # Calculate transitions across all year pairs in decade
        decade_trans = np.zeros((len(present_classes), len(present_classes)))
        
        for i in range(len(decade_indices)-1):
            from_data = yearly_data[decade_indices[i]]
            to_data = yearly_data[decade_indices[i+1]]
            
            for from_idx, from_cls in enumerate(present_classes):
                mask = (from_data == from_cls)
                if np.any(mask):
                    to_values = to_data[mask]
                    unique_to, counts = np.unique(to_values, return_counts=True)
                    for to_cls, cnt in zip(unique_to, counts):
                        if to_cls in present_classes:
                            to_idx = present_classes.index(to_cls)
                            decade_trans[from_idx, to_idx] += cnt
        
        # Remove self-transitions
        np.fill_diagonal(decade_trans, 0)
        
        # Save change matrix
        change_matrix_csv = os.path.join(output_dir, f'change_matrix_{start_year}_{end_year}.csv')
        with open(change_matrix_csv, 'w') as f:
            f.write("From_Class,From_Label,To_Class,To_Label,Count\n")
           
            for j, to_cls in enumerate(present_classes):
                if i < decade_trans.shape[0] and j < decade_trans.shape[1]:
                    count = decade_trans[i, j]
                else:
                    logging.warning(f"Skipping out-of-bounds index: i={i}, j={j}")
                    count = 0
                if count > 0:
                    f.write(f"{from_cls},{LABELS[from_cls]},{to_cls},{LABELS[to_cls]},{count}\n")
        
        # Create a graphical table of the change matrix
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = [["From/To"] + [LABELS[to_cls] for to_cls in present_classes]]
        for i, from_cls in enumerate(present_classes):
            row = [LABELS[from_cls]] + [int(decade_trans[i, j]) for j in range(len(present_classes))]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', colLoc='center')

        # Apply colors to the table cells
        for i, row in enumerate(table_data[1:], start=1):  # Skip header row
            for j, cell_value in enumerate(row[1:], start=1):  # Skip first column
                if i < j:  # Above the main diagonal (columns)
                    cls = present_classes[j - 1]
                    if cls in COLOR_MAP:
                        color = COLOR_MAP[cls]
                        table[(i, j)].set_facecolor(color)
                        # Set text color based on brightness
                        brightness = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color))[2]
                        table[(i, j)].get_text().set_color('white' if brightness < 0.6 else 'black')
                elif i > j:  # Below the main diagonal (rows)
                    cls = present_classes[i - 1]
                    if cls in COLOR_MAP:
                        color = COLOR_MAP[cls]
                        table[(i, j)].set_facecolor(color)
                        # Set text color based on brightness
                        brightness = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color))[2]
                        table[(i, j)].get_text().set_color('white' if brightness < 0.6 else 'black')
        for i, row in enumerate(table_data[1:], start=1):  # Skip header row
            for j, cell_value in enumerate(row[1:], start=1):  # Skip first column
                if i < j:  # Above the main diagonal (columns)
                    cls = present_classes[j - 1]
                    if cls in COLOR_MAP:
                        table[(i, j)].set_facecolor(COLOR_MAP[cls])
                elif i > j:  # Below the main diagonal (rows)
                    cls = present_classes[i - 1]
                    if cls in COLOR_MAP:
                        table[(i, j)].set_facecolor(COLOR_MAP[cls])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(table_data[0]))))
        
        # Save the table as an image
        table_image_path = os.path.join(output_dir, f'change_matrix_{start_year}_{end_year}.png')
        plt.savefig(table_image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create Sankey diagram
        title = f"{grid_name} Land Cover Changes {start_year}-{end_year}"
        output_html = os.path.join(output_dir, f'new_transitions_{start_year}_{end_year}.html')
        output_csv = os.path.join(output_dir, f'new_transitions_{start_year}_{end_year}.csv')
        
        create_sankey(
            transition_matrix=decade_trans,
            classes=present_classes,
            title=title,
            output_html=output_html,
            output_csv=output_csv
        )
    
    # Create full-period diagram (first to last year)
    if len(yearly_data) >= 2:
        full_trans = np.zeros((len(present_classes), len(present_classes)))
        from_data = yearly_data[0]
        to_data = yearly_data[-1]
        
        for from_idx, from_cls in enumerate(present_classes):
            mask = (from_data == from_cls)
            if np.any(mask):
                to_values = to_data[mask]
                unique_to, counts = np.unique(to_values, return_counts=True)
                for to_cls, cnt in zip(unique_to, counts):
                    if to_cls in present_classes:
                        to_idx = present_classes.index(to_cls)
                        full_trans[from_idx, to_idx] += cnt
        
        # Remove self-transitions
        np.fill_diagonal(full_trans, 0)
        
        title_full = f"{grid_name} Land Cover Changes {years[0]}-{years[-1]}"
        output_html_full = os.path.join(output_dir, 'transitions_full.html')
        output_csv_full = os.path.join(output_dir, 'transitions_full.csv')
        
        create_sankey(
            transition_matrix=full_trans,
            classes=present_classes,
            title=title_full,
            output_html=output_html_full,
            output_csv=output_csv_full
        )
def create_decadal_sankey_diagrams(root, output_dir):
    """Create Sankey diagrams showing both persistence and transitions with proportional sizing."""
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    try:
        if 'changes' in root:
            changes_data = root['changes'][:]
        else:
            raise ValueError("Missing 'changes' dataset in Zarr root")
            
        decadal_windows = [(1985, 1995), (1995, 2005), (2005, 2015), (2015, 2023)]
        
        # Get all classes from the data, excluding 0 (no data)
        all_classes = sorted(set(np.unique(changes_data)) & set(LABELS.keys()))
        all_classes = [cls for cls in all_classes if cls != 0]
        
        # Prepare labels and colors
        class_labels = {cls: LABELS.get(cls, f"Class {cls}") for cls in all_classes}
        class_colors = {cls: matplotlib.colors.to_rgb(COLOR_MAP.get(cls, "#999999")) 
                       for cls in all_classes if cls in COLOR_MAP}
        
        # Add default colors for missing classes
        for cls in all_classes:
            if cls not in class_colors:
                class_colors[cls] = matplotlib.colors.to_rgb("#999999")

        for start_year, end_year in decadal_windows:
            try:
                # Calculate transition matrix (including self-transitions)
                transition_matrix = np.zeros((len(all_classes), len(all_classes)))
                
                # Map class IDs to matrix indices
                class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
                
                # Populate transition matrix
                # for val in changes_data.flatten():
                #     if val != 0:  # Skip no data
                #         from_cls, to_cls = val, val  # Default to self-transition
                #         transition_matrix[class_to_idx[from_cls], class_to_idx[to_cls]] += 1
                for from_cls, to_cls in changes_data.reshape(-1, 2):
                    if from_cls in LABELS and to_cls in LABELS:
                        if from_cls != 0 and to_cls != 0:  # Optional: filter out "no data"
                            transition_matrix[class_to_idx[from_cls], class_to_idx[to_cls]] += 1

                
                # Calculate node sizes (sum of incoming and outgoing flows)
                out_flows = np.sum(transition_matrix, axis=1)  # Sum of outgoing flows per class
                in_flows = np.sum(transition_matrix, axis=0)    # Sum of incoming flows per class
                
                # Create node structure (left and right sides)
                node_names = []
                node_colors = []
                node_positions = {}
                
                # Left nodes (source)
                for i, cls in enumerate(all_classes):
                    label = f"{cls}: {class_labels[cls]} (Start)"
                    node_names.append(label)
                    node_positions[(cls, 'start')] = i
                    rgb = class_colors[cls]
                    node_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)")
                
                # Right nodes (target)
                right_offset = len(all_classes)
                for i, cls in enumerate(all_classes):
                    label = f"{cls}: {class_labels[cls]} (End)"
                    node_names.append(label)
                    node_positions[(cls, 'end')] = right_offset + i
                    rgb = class_colors[cls]
                    node_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)")
                
                # Calculate node positions proportional to flow sizes
                def calc_positions(flows):
                    total = sum(flows)
                    if total == 0:
                        return [0.5] * len(flows)  # Fallback if no flows
                    positions = []
                    cumulative = 0
                    for flow in flows:
                        positions.append((cumulative + flow/2) / total)
                        cumulative += flow
                    return positions
                
                left_y = calc_positions(out_flows)
                right_y = calc_positions(in_flows)
                
                # Combine positions
                node_x = [0.1] * len(all_classes) + [0.9] * len(all_classes)
                node_y = left_y + right_y
                
                # Create links (include all transitions)
                sources = []
                targets = []
                values = []
                link_colors = []
                
                for i, from_cls in enumerate(all_classes):
                    for j, to_cls in enumerate(all_classes):
                        value = transition_matrix[i, j]
                        if value > 0:
                            sources.append(node_positions[(from_cls, 'start')])
                            targets.append(node_positions[(to_cls, 'end')])
                            values.append(int(value))
                            # Use source color for links
                            rgb = class_colors[from_cls]
                            alpha = 0.4 if i == j else 0.6  # Lighter for self-transitions
                            link_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})")
                
                # Create the Sankey diagram
                fig = go.Figure(go.Sankey(
                    arrangement="fixed",
                    node=dict(
                        pad=30,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=node_names,
                        color=node_colors,
                        x=node_x,
                        y=node_y,
                        customdata=[f"Out: {out_flows[i]:,}<br>In: {in_flows[i]:,}" 
                                  for i in range(len(all_classes))] * 2,
                        hovertemplate="%{label}<br>%{customdata}<extra></extra>"
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                        customdata=[f"{all_classes[i//len(all_classes)]} → {all_classes[j%len(all_classes)]}"
                                   for i, j in zip(sources, targets)],
                        hovertemplate="%{customdata}<br>Count: %{value:,}<extra></extra>"
                    )
                ))
                
                # Update layout
                fig.update_layout(
                    title_text=f"{grid_name} Land Cover Changes {start_year}-{end_year}",
                    font=dict(size=12, family="Arial"),
                    height=max(1200, len(all_classes) * 60),
                    width=1600,
                    margin=dict(l=150, r=150, b=100, t=120, pad=20)
                )
                
                # Save outputs
                html_path = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.html')
                plot(fig, filename=html_path, auto_open=False, include_plotlyjs='cdn')
                logging.info(f"Created diagram for {start_year}-{end_year}")
                
            except Exception as e:
                logging.error(f"Error creating {start_year}-{end_year} diagram: {str(e)}", exc_info=True)
                
    except Exception as e:
        logging.error(f"Error in decadal Sankey diagrams: {str(e)}", exc_info=True)
        raise

def create_sankey(transition_matrix, classes, title, output_html, output_csv):
    """Create a Sankey diagram with properly aligned nodes and proportional sizing."""
    
    # Calculate total flows for positioning
    total_flow = np.sum(transition_matrix)
    if total_flow == 0:
        logging.warning(f"No valid flows found for {title}")
        return
    
    # Calculate node sizes (sum of incoming and outgoing flows)
    out_flows = np.sum(transition_matrix, axis=1)  # Sum of outgoing flows per class
    in_flows = np.sum(transition_matrix, axis=0)   # Sum of incoming flows per class
    
    # Calculate node positions proportional to flow sizes
    def calc_positions(flows):
        total = sum(flows)
        if total == 0:
            return [0.5] * len(flows)  # Fallback if no flows
        positions = []
        cumulative = 0
        for flow in flows:
            positions.append((cumulative + flow/2) / total)
            cumulative += flow
        return positions
    
    left_y = calc_positions(out_flows)  # Left nodes positioned by outgoing flow
    right_y = calc_positions(in_flows)  # Right nodes positioned by incoming flow
    
    # Create node structure
    node_x = [0.1] * len(classes) + [0.9] * len(classes)  # Left and right columns
    node_y = left_y + right_y
    
    # Prepare link data (include all significant flows)
    sources, targets, values, link_colors = [], [], [], []
    threshold = max(total_flow * 0.001, 10)  # Minimum threshold
    
    for i, from_cls in enumerate(classes):
        for j, to_cls in enumerate(classes):
            value = transition_matrix[i, j]
            if value > threshold:
                sources.append(i)
                targets.append(j + len(classes))  # Target nodes offset
                values.append(value)
                # Use source color with different alpha for self-transitions
                base_color = COLOR_MAP.get(from_cls, '#999999')
                alpha = 0.4 if i == j else 0.7  # Lighter for self-transitions
                link_colors.append(f"rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, {alpha})")
    
    # Create node labels with flow information
    node_labels = [
        f"{cls}: {LABELS.get(cls, '?')}<br>Out: {out_flows[i]:,}"
        for i, cls in enumerate(classes)
    ] + [
        f"{cls}: {LABELS.get(cls, '?')}<br>In: {in_flows[i]:,}"
        for i, cls in enumerate(classes)
    ]
    
    # Node colors (same for left and right)
    node_colors = [COLOR_MAP.get(cls, '#999999') for cls in classes]
    
    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=10,
            thickness=10,
            line=dict(color="black", width=0.3),
            label=node_labels,
            color=node_colors,
            x=node_x,
            y=node_y,
            hovertemplate="%{label}<extra></extra>"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="From: %{source.label}<br>To: %{target.label}<br>Count: %{value:,}<extra></extra>"
        )
    ))
    
    # Update layout with dynamic sizing
    fig.update_layout(
        title_text=title,
        font=dict(size=10, family="Arial"),
        height=max(1000, len(classes) * 60),  # Dynamic height based on class count
        width=1600,
        margin=dict(l=100, r=100, b=50, t=80, pad=10)
    )
    
    # Save outputs
    plot(fig, filename=output_html, auto_open=False)
    
    # Save CSV with all transitions
    with open(output_csv, 'w') as f:
        f.write("From_Class,From_Label,To_Class,To_Label,Count,Percent\n")
        for i, from_cls in enumerate(classes):
            for j, to_cls in enumerate(classes):
                value = transition_matrix[i, j]
                if value > 0:
                    percent = 100 * value / total_flow
                    f.write(
                        f"{from_cls},{LABELS.get(from_cls, '?')},"
                        f"{to_cls},{LABELS.get(to_cls, '?')},"
                        f"{value},{percent:.2f}%\n"
                    )
    
    logging.info(f"Created Sankey diagram: {output_html}")

if __name__ == '__main__':
    VRT_FILE = '/srv/extrassd/2025_mapbiomas/mapbiomas_coverage_1985_2023.vrt'
    OUTPUT_BASE_DIR = 'grid_results_robust'


    
    # Example 10x10 degree polygon
    POLYGON_8x8 = [((-42, -16), (-38, -16), (-38, -22), (-38, -16), (-42, -16))]
    
    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        logging.info("Starting robust analysis with visualization")
        
        zarr_path, grid_output_dir = extract_grid_data_optimized(VRT_FILE, POLYGON_8x8, OUTPUT_BASE_DIR)
        create_visualizations(zarr_path, grid_output_dir)
        
        
        logging.info(f"Analysis complete. Results saved to {grid_output_dir}")
        
    except Exception as e:
        logging.critical(f"Analysis failed: {str(e)}", exc_info=True)