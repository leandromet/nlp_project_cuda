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
    30: "#9c0027", 31: "#091077", 32: "#fc8114", 33: "#2532e4", 35: "#9065d0", 36:"#d082de",
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
    #create_landcover_map(root, output_dir)
    
    # 2. Create changes map
   # create_changes_map(root, output_dir)
    
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
    else:
        yearly_data = [np.asarray(root['first_year'][:]), np.asarray(root['last_year'][:])]
        years = [1985, 2023]
    
    # Define decades
    decades = [
        (1985, 1994),  # 1985-1994 (10 years)
        (1995, 2004),  # 1995-2004 (10 years)
        (2005, 2014),  # 2005-2014 (10 years)
        (2015, 2023)   # 2015-2023 (9 years)
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
                count = decade_trans[i, j]
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
        output_html = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.html')
        output_csv = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.csv')
        
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
    """Create Sankey diagrams ignoring class 0 (no data) with value-proportional nodes."""
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    try:
        if 'changes' in root:
            changes_data = root['changes'][:]
        else:
            raise ValueError("Missing 'changes' dataset in Zarr root")
            
        decadal_windows = [(1985, 1995), (1995, 2005), (2005, 2015), (2015, 2023)]
        
        # Get all possible classes from the data, excluding 0 (no data)
        all_possible_classes = sorted(set(np.unique(changes_data)))
        all_possible_classes = [cls for cls in all_possible_classes if cls != 0]  # Exclude class 0
        
        class_labels = {cls: LABELS.get(cls, f"Class {cls}") for cls in all_possible_classes}
        class_colors = {cls: matplotlib.colors.to_rgb(COLOR_MAP.get(cls, "#999999")) 
                               for cls in all_possible_classes if cls in COLOR_MAP}
        
        # Add default colors for missing classes
        for cls in all_possible_classes:
            if cls not in class_colors:
                class_colors[cls] = matplotlib.colors.to_rgb("#999999")

        for start_year, end_year in decadal_windows:
            try:
                # Calculate persistence (unchanged pixels), excluding class 0
                persistence_mask = (changes_data == 0)
                persisted_classes, persisted_counts = np.unique(changes_data[persistence_mask], 
                                                              return_counts=True)
                # Filter out class 0 from persisted classes
                persisted_classes = [cls for cls in persisted_classes if cls != 0]
                persisted_counts = [cnt for cls, cnt in zip(persisted_classes, persisted_counts) if cls != 0]
                
                # Calculate transitions (changed pixels), excluding transitions involving class 0
                transition_mask = (changes_data > 0)
                transition_pairs, transition_counts = np.unique(
                    np.vstack((changes_data[transition_mask], changes_data[transition_mask])).T,
                    axis=0,
                    return_counts=True
                )
                # Filter out transitions involving class 0
                valid_transitions = [(f, t) for f, t in transition_pairs if f != 0 and t != 0]
                transition_counts = [cnt for (f, t), cnt in zip(transition_pairs, transition_counts) 
                                   if f != 0 and t != 0]
                
                # Calculate total flow for each class (incoming + outgoing)
                class_flows = defaultdict(float)
                
                # Add persistence flows (already filtered)
                for cls, count in zip(persisted_classes, persisted_counts):
                    class_flows[cls] += count
                
                # Add transition flows (already filtered)
                for (from_cls, to_cls), count in zip(valid_transitions, transition_counts):
                    class_flows[from_cls] += count
                    class_flows[to_cls] += count
                
                # Sort classes by total flow (descending)
                sorted_classes = sorted(class_flows.keys(), key=lambda x: -class_flows[x])
                num_classes = len(sorted_classes)
                
                # Create node structure
                node_names = []
                node_colors = []
                node_positions = {}
                node_values = []
                
                # Create nodes for both sides
                for side in ['start', 'end']:
                    for i, cls in enumerate(sorted_classes):
                        label = f"{cls}: {class_labels.get(cls, f'Unknown Class {cls}')} ({side.capitalize()})"
                        node_names.append(label)
                        node_positions[(cls, side)] = i if side == 'start' else i + num_classes
                        rgb = class_colors[cls]
                        node_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)")
                        node_values.append(class_flows[cls])
                
                # Calculate cumulative flows for positioning
                total_flow = sum(class_flows.values())
                if total_flow == 0:
                    logging.warning(f"No valid transitions found for {start_year}-{end_year}")
                    continue
                
                cumulative_flow = 0
                y_positions = []
                
                for cls in sorted_classes:
                    # Calculate center position for this node
                    y_pos = (cumulative_flow + class_flows[cls]/2) / total_flow
                    y_positions.append(y_pos)
                    cumulative_flow += class_flows[cls]
                
                # Apply to both sides
                node_y = y_positions + y_positions
                node_x = [0.1] * num_classes + [0.9] * num_classes
                
                # Create links
                sources = []
                targets = []
                values = []
                link_colors = []
                
                # Add persistence flows
                for cls, count in zip(persisted_classes, persisted_counts):
                    if (cls, 'start') in node_positions and (cls, 'end') in node_positions:
                        sources.append(node_positions[(cls, 'start')])
                        targets.append(node_positions[(cls, 'end')])
                        values.append(int(count))
                        rgb = class_colors[cls]
                        link_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.4)")
                
                # Add transition flows
                for (from_cls, to_cls), count in zip(valid_transitions, transition_counts):
                    if (from_cls, 'start') in node_positions and (to_cls, 'end') in node_positions:
                        sources.append(node_positions[(from_cls, 'start')])
                        targets.append(node_positions[(to_cls, 'end')])
                        values.append(int(count))
                        rgb = class_colors.get(from_cls, (0.6, 0.6, 0.6))
                        link_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.6)")
                
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
                        customdata=node_values,
                        hovertemplate="%{label}<br>Total Flow: %{customdata:,}<extra></extra>"
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors
                    )
                ))
                
                # Update layout
                fig.update_layout(
                    title_text=f"{grid_name} Land Cover Changes {start_year}-{end_year}<br>"
                             f"(Excluding No Data, Node heights proportional to flow)",
                    font=dict(size=12, family="Arial"),
                    height=max(1200, num_classes * 60),
                    width=1600,
                    margin=dict(l=150, r=150, b=100, t=120, pad=20)
                )
                
                # Save outputs
                html_path = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.html')
                png_path = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.png')
                
                plot(fig, filename=html_path, auto_open=False, include_plotlyjs='cdn')
                
                try:
                    fig.write_image(png_path, scale=2, engine="kaleido")
                    logging.info(f"Saved diagram to {png_path}")
                except Exception as e:
                    logging.warning(f"Could not save static image: {str(e)}")
                
            except Exception as e:
                logging.error(f"Error creating {start_year}-{end_year} diagram: {str(e)}", exc_info=True)
                
    except Exception as e:
        logging.error(f"Error in decadal Sankey diagrams: {str(e)}", exc_info=True)
        raise


def create_sankey(transition_matrix, classes, title, output_html, output_csv):
    """Create a Sankey diagram with proper left-right separation."""
    # Calculate total flow for threshold
    total_flow = np.sum(transition_matrix)
    threshold = max(total_flow * 0.001, 10)  # Minimum 10 pixels or 0.1% of flow
    
    # Prepare transitions - ensure targets point to right-side nodes
    sources, targets, values = [], [], []
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = transition_matrix[i, j]
            if value > threshold and i != j:
                sources.append(i)
                targets.append(j + len(classes))  # CRITICAL: Offset by len(classes)
                values.append(value)
    
    # Calculate node positions - ensure right nodes stay right
    num_nodes = len(classes)
    node_x = [0.1] * num_nodes + [0.9] * num_nodes  # Left and right columns
    node_y_left = np.linspace(0.1, 0.9, num_nodes)  # Evenly spaced left nodes
    node_y_right = np.linspace(0.1, 0.9, num_nodes)  # Evenly spaced right nodes
    
    # Adjust right positions based on actual outflows
    outflows = np.sum(transition_matrix, axis=0)
    if sum(outflows) > 0:
        node_y_right = np.cumsum(outflows) / sum(outflows) * 0.8 + 0.1
    
    node_y = np.concatenate([node_y_left, node_y_right])
    
    # Node labels with counts
    left_counts = np.sum(transition_matrix, axis=1)
    right_counts = np.sum(transition_matrix, axis=0)
    node_labels = [
        f"{cls}: {LABELS.get(cls, '?')} ({left_counts[i]:,})" 
        for i, cls in enumerate(classes)
    ] + [
        f"{cls}: {LABELS.get(cls, '?')} ({right_counts[j]:,})" 
        for j, cls in enumerate(classes)
    ]
    
    # Create figure
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=30,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=[COLOR_MAP.get(cls, '#999999') for cls in classes * 2],
            x=node_x,
            y=node_y.tolist()
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=[COLOR_MAP.get(classes[i], '#999999') for i in sources]
        )
    ))
    
    fig.update_layout(
        title_text=title,
        font_size=10,
        height=max(800, len(classes) * 60),
        width=1400,
        margin=dict(l=150, r=150, b=100, t=100)
    )
    
    plot(fig, filename=output_html, auto_open=False)
    
    # Save CSV with all transitions (not just filtered ones)
    with open(output_csv, 'w') as f:
        f.write("From_Class,From_Label,To_Class,To_Label,Count,Percent\n")
        for i in range(len(classes)):
            for j in range(len(classes)):
                value = transition_matrix[i, j]
                if value > 0:
                    percent = 100 * value / total_flow
                    f.write(
                        f"{classes[i]},{LABELS.get(classes[i], '?')},"
                        f"{classes[j]},{LABELS.get(classes[j], '?')},"
                        f"{value},{percent:.2f}%\n"
                    )
    
    logging.info(f"Created Sankey diagram: {output_html}")


if __name__ == '__main__':
    VRT_FILE = '/srv/extrassd/2025_mapbiomas/mapbiomas_coverage_1985_2023.vrt'
    OUTPUT_BASE_DIR = 'grid_results_robust'


    
    # Example 10x10 degree polygon
    POLYGON_8x8 = [((-40, -19), (-38, -19), (-38, -22), (-40, -22), (-40, -19))]
    
    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        logging.info("Starting robust analysis with visualization")
        
        zarr_path, grid_output_dir = extract_grid_data_optimized(VRT_FILE, POLYGON_8x8, OUTPUT_BASE_DIR)
        create_visualizations(zarr_path, grid_output_dir)
        
        
        logging.info(f"Analysis complete. Results saved to {grid_output_dir}")
        
    except Exception as e:
        logging.critical(f"Analysis failed: {str(e)}", exc_info=True)