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
   # create_decadal_sankey_diagrams(root, output_dir)
    # 5. Create land cover maps for decades
    create_landcover_maps_for_decades(root, output_dir)
    # 6. Create changes maps for decades
    create_changes_maps_for_decades(root, output_dir)


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

def create_landcover_maps_for_decades(root, output_dir):
    """Create land cover maps for the start year of each working decade."""
    grid_name = root.attrs['grid_name']
    transform = rasterio.transform.Affine(*json.loads(root.attrs['window_transform']))
    
    # Create colormap
    cmap = ListedColormap([COLOR_MAP.get(i, '#ffffff') for i in range(max(COLOR_MAP.keys()) + 1)])
    
    # Define decades
    decades = [
        (1985, 1995),  # 1985-1994
        (1995, 2005),  # 1995-2004
        (2005, 2015),  # 2005-2014
        (2015, 2023)   # 2015-2023
    ]
    
    for start_year, _ in decades:
        band_index = start_year - 1985  # Calculate band index based on start year
        if band_index < 0 or band_index >= root['data'].shape[0]:
            logging.warning(f"Start year {start_year} is out of range for available data.")
            continue
        
        start_year_data = root['data'][band_index][:]
        
        plt.figure(figsize=(12, 20))
        plt.imshow(start_year_data, cmap=cmap, vmin=0, vmax=max(COLOR_MAP.keys()))
        plt.title(f"{grid_name} Land Cover {start_year}")
        
        # Create legend
        from matplotlib.patches import Patch
        patches = [Patch(color=COLOR_MAP[k], label=f"{k}: {LABELS[k]}") 
                   for k in sorted(LABELS.keys()) if k in COLOR_MAP]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'landcover_map_{start_year}.png')
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

def create_changes_maps_for_decades(root, output_dir):
    """Create changes frequency maps for the first year of each decade and the last year of data."""
    grid_name = root.attrs['grid_name']
    changes = root['changes'][:]
    transform = json.loads(root.attrs['window_transform'])
    
    # Define decades and the last year
    years = list(range(1985, 1985 + changes.shape[0]))
    decades = [1985 + i * 10 for i in range((years[-1] - 1985) // 10 + 1)]
    decades.append(years[-1])  # Add the last year of data
    
    for year in decades:
        # Get the changes for the specific year
        year_index = year - 1985
        if year_index < 0 or year_index >= changes.shape[0]:
            logging.warning(f"Year {year} is out of range for available data.")
            continue
        
        year_changes = changes[year_index]
        
        # Create the changes map
        plt.figure(figsize=(12, 20))
        plt.imshow(year_changes, cmap='gist_stern')
        plt.colorbar(label=f'Number of Changes ({year})')
        plt.title(f"{grid_name} Change Frequency {year}")
        
        # Save the PNG file
        output_path = os.path.join(output_dir, f'changes_map_{year}.png')
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
        (2015, 2023),  # 2015-2023 (9 years)
        (1985, 2023)   # Full period (38 years)
    ]

    all_classes = sorted(set(LABELS.keys()) - {0})  # Optional: exclude 'No data' (0)

    for start_year, end_year in decades:
        decade_indices = [i for i, y in enumerate(years) if start_year <= y <= end_year]
        if len(decade_indices) < 2:
            continue  # Skip if not enough data
        
        transition_matrix = defaultdict(lambda: defaultdict(int))
        observed_starts = set()
        observed_ends = set()
        total_count = 0
        
        for i in range(len(decade_indices) - 1):
            from_data = yearly_data[decade_indices[i]]
            to_data = yearly_data[decade_indices[i + 1]]
            
            unique_transitions, counts = np.unique(
                np.stack((from_data.flatten(), to_data.flatten()), axis=1), axis=0, return_counts=True
            )
            for (from_cls, to_cls), count in zip(unique_transitions, counts):
                if from_cls in LABELS and to_cls in LABELS:
                    transition_matrix[from_cls][to_cls] += count
                    total_count += count
        
        threshold = max(total_count * 0.001, 10)

        # Build final class sets after filtering by threshold
        for from_cls, to_dict in transition_matrix.items():
            for to_cls, value in to_dict.items():
                if value > threshold:
                    observed_starts.add(from_cls)
                    observed_ends.add(to_cls)

        start_classes = [cls for cls in all_classes if cls in observed_starts]
        end_classes = [cls for cls in all_classes if cls in observed_ends]
        
        # Map class IDs to Sankey node indices
        source_idx_map = {cls: i for i, cls in enumerate(start_classes)}
        target_idx_map = {cls: i + len(start_classes) for i, cls in enumerate(end_classes)}

        # Build links
        sources, targets, values, link_colors = [], [], [], []

        for from_cls in start_classes:
            for to_cls in end_classes:
                value = transition_matrix[from_cls][to_cls]
                if value > threshold:
                    sources.append(source_idx_map[from_cls])
                    targets.append(target_idx_map[to_cls])
                    values.append(value)
                    base_color = COLOR_MAP.get(from_cls, '#999999')
                    alpha = 0.4 if from_cls == to_cls else 0.7
                    link_colors.append(
                        f"rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, {alpha})"
                    )

        # Node labels and colors
        labels = (
            [f"{LABELS[cls]}" for cls in start_classes] +
            [f"{LABELS[cls]}" for cls in end_classes]
            # Optionally use suffixes:
            # [f"{LABELS[cls]} (from)" for cls in start_classes] +
            # [f"{LABELS[cls]} (to)" for cls in end_classes]
        )
        node_colors = [COLOR_MAP.get(cls, '#999999') for cls in start_classes + end_classes]

        fig = go.Figure(go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=30,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors,
                x=[0.1] * len(start_classes) + [0.9] * len(end_classes),
                y=list(np.linspace(0, 1, len(start_classes))) + list(np.linspace(0, 1, len(end_classes))),
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

        fig.update_layout(
            title_text=f"{grid_name} Land Cover Changes {start_year}-{end_year}",
            font=dict(size=12, family="Arial"),
            height=max(1200, len(start_classes + end_classes) * 60),
            width=1600,
            margin=dict(l=150, r=150, b=100, t=120, pad=20)
        )

        html_path = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.html')
        plot(fig, filename=html_path, auto_open=False, include_plotlyjs='cdn')
        logging.info(f"Created diagram for {start_year}-{end_year}")

    """Create Sankey diagrams with proper node alignment and proportional flow visualization."""
    grid_name = root.attrs['grid_name']
    
    # Get data as numpy arrays
    if 'data' in root:
        yearly_data = [np.asarray(root['data'][i][:]) for i in range(root['data'].shape[0])]
        years = list(range(1985, 1985 + len(yearly_data)))
    else:
        yearly_data = [np.asarray(root['first_year'][:]), np.asarray(root['last_year'][:])]
        years = [1985, 2023]
    
    # Define analysis periods
    periods = [
        (1985, 1995),  # First decade
        (1995, 2005),  # Second decade
        (2005, 2015),  # Third decade
        (2015, 2023),  # Recent period
        (1985, 2023)   # Full period
    ]
    
    for start_year, end_year in periods:
        period_indices = [i for i, y in enumerate(years) if start_year <= y <= end_year]
        if len(period_indices) < 2:
            continue  # Skip if not enough data
        
        # Calculate all transitions during this period
        transition_counts = defaultdict(lambda: defaultdict(int))
        start_classes = set()
        end_classes = set()
        
        for i in range(len(period_indices) - 1):
            from_data = yearly_data[period_indices[i]]
            to_data = yearly_data[period_indices[i + 1]]
            
            # Find all transitions between classes
            unique_pairs, counts = np.unique(
                np.stack([from_data.flatten(), to_data.flatten()], axis=1),
                axis=1, return_counts=True
            )
            
            for (from_cls, to_cls), count in zip(unique_pairs, counts):
                if from_cls in LABELS and to_cls in LABELS:  # Only include labeled classes
                    transition_counts[from_cls][to_cls] += count
                    start_classes.add(from_cls)
                    end_classes.add(to_cls)
        
        if not start_classes or not end_classes:
            continue  # Skip if no valid transitions found
        
        # Prepare all classes (union of start and end classes)
        all_classes = sorted(start_classes.union(end_classes))
        class_idx = {cls: i for i, cls in enumerate(all_classes)}
        
        # Calculate node sizes (total incoming and outgoing flows)
        out_flows = defaultdict(int)
        in_flows = defaultdict(int)
        
        for from_cls in transition_counts:
            for to_cls, count in transition_counts[from_cls].items():
                out_flows[from_cls] += count
                in_flows[to_cls] += count
        
        # Calculate node positions proportional to flow sizes
        def calc_positions(flows_dict, classes):
            total = sum(flows_dict.get(cls, 0) for cls in classes)
            if total == 0:
                return [i/(len(classes)+1) for i in range(1, len(classes)+1)]  # Even spacing fallback
            
            positions = []
            cumulative = 0
            for cls in classes:
                flow = flows_dict.get(cls, 0)
                positions.append((cumulative + flow/2) / total)
                cumulative += flow
            return positions
        
        left_y = calc_positions(out_flows, all_classes)
        right_y = calc_positions(in_flows, all_classes)
        
        # Create links
        sources, targets, values, link_colors = [], [], [], []
        total_flow = sum(out_flows.values())
        threshold = max(total_flow * 0.001, 10)  # Minimum threshold
        
        for from_cls in transition_counts:
            for to_cls, count in transition_counts[from_cls].items():
                if count > threshold:
                    sources.append(class_idx[from_cls])
                    targets.append(class_idx[to_cls] + len(all_classes))  # Offset for right side
                    values.append(count)
                    base_color = COLOR_MAP.get(from_cls, '#999999')
                    alpha = 0.4 if from_cls == to_cls else 0.7  # Lighter for self-transitions
                    link_colors.append(
                        f"rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, {alpha})"
                    )
        
        # Create node labels with flow information
        node_labels = [
            f"{cls}: {LABELS[cls]}<br>Out: {out_flows.get(cls, 0):,}"
            for cls in all_classes
        ] + [
            f"{cls}: {LABELS[cls]}<br>In: {in_flows.get(cls, 0):,}"
            for cls in all_classes
        ]
        
        # Create the Sankey diagram
        fig = go.Figure(go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=30,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=[COLOR_MAP.get(cls, '#999999') for cls in all_classes * 2],
                x=[0.1] * len(all_classes) + [0.9] * len(all_classes),
                y=left_y + right_y,
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
        
        # Update layout
        fig.update_layout(
            title_text=f"{grid_name} Land Cover Changes {start_year}-{end_year}",
            font=dict(size=12, family="Arial"),
            height=max(1200, len(all_classes) * 60),
            width=1600,
            margin=dict(l=200, r=200, b=100, t=120, pad=20)
        )
        
        # Save outputs
        html_path = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.html')
        plot(fig, filename=html_path, auto_open=False, include_plotlyjs='cdn')
        
        
        csv_path = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.csv')
        with open(csv_path, 'w') as f:
            f.write("From_Class,From_Label,To_Class,To_Label,Count,Percent\n")
            for from_cls in transition_counts:
                for to_cls, count in transition_counts[from_cls].items():
                    percent = 100 * count / total_flow if total_flow > 0 else 0
                    f.write(
                        f"{from_cls},{LABELS[from_cls]},{to_cls},{LABELS[to_cls]},"
                        f"{count},{percent:.2f}%\n"
                    )
        
        logging.info(f"Created diagram for {start_year}-{end_year}")

        # Save transition matrix as PNG
        transition_matrix = np.zeros((len(all_classes), len(all_classes)), dtype=np.uint64)
        for from_cls in transition_counts:
            for to_cls, count in transition_counts[from_cls].items():
                transition_matrix[class_idx[from_cls], class_idx[to_cls]] = count

        plt.figure(figsize=(12, 12))
        plt.imshow(transition_matrix, cmap=ListedColormap([COLOR_MAP.get(i, '#ffffff') for i in range(max(COLOR_MAP.keys()) + 1)]), interpolation='nearest')
        
        # Add text labels to the matrix
        for i in range(len(all_classes)):
            for j in range(len(all_classes)):
                value = transition_matrix[i, j]
            if value > 0:
                color = COLOR_MAP.get(all_classes[i], '#ffffff')
                # Determine text color based on background brightness
                brightness = (int(color[1:3], 16) * 0.299 + int(color[3:5], 16) * 0.587 + int(color[5:7], 16) * 0.114) / 255
                text_color = 'white' if brightness < 0.5 else 'black'
                plt.text(j, i, f"{value}", ha='center', va='center', fontsize=6, color=text_color)
        plt.colorbar(label='Transition Count')
        plt.xticks(ticks=np.arange(len(all_classes)), labels=[LABELS[cls] for cls in all_classes], rotation=90, fontsize=8, color='black')
        plt.yticks(ticks=np.arange(len(all_classes)), labels=[LABELS[cls] for cls in all_classes], fontsize=8, color='black')
        plt.title(f"Transition Matrix {start_year}-{end_year}", fontsize=14, color='black')
        plt.tight_layout()

        png_path = os.path.join(output_dir, f'transition_matrix_{start_year}_{end_year}.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    VRT_FILE = '/srv/extrassd/2025_mapbiomas/mapbiomas_coverage_1985_2023.vrt'
    OUTPUT_BASE_DIR = 'grid_results_robust'


    
    # Example 10x10 degree polygon
    POLYGON_8x8 = [((-41, -21), (-41, -19), (-40, -19), (-40, -21), (-41, -21))]
    
    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        logging.info("Starting robust analysis with visualization")
        
        zarr_path, grid_output_dir = extract_grid_data_optimized(VRT_FILE, POLYGON_8x8, OUTPUT_BASE_DIR)
        create_visualizations(zarr_path, grid_output_dir)
        
        
        logging.info(f"Analysis complete. Results saved to {grid_output_dir}")
        
    except Exception as e:
        logging.critical(f"Analysis failed: {str(e)}", exc_info=True)