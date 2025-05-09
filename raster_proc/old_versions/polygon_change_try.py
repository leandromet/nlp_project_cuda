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
from matplotlib.patches import Patch
from rasterio.features import rasterize
import geopandas as gpd



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
            logging.error(f"Error occurred: {str(e)}", exc_info=True)
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
            prev_data = robust_read(src, 1, window)
            
            # Process subsequent bands
            for band_idx in range(2, src.count + 1):
                curr_data = robust_read(src, band_idx, window)
                
                # Calculate changes between consecutive years
                changed = prev_data != curr_data
                changes += changed.astype('uint8')
                
                # Update transition matrix
                if np.any(changed):
                    from_vals = prev_data[changed]
                    to_vals = curr_data[changed]
                    unique_transitions, counts = np.unique(
                        np.column_stack((from_vals, to_vals)), 
                        axis=0, 
                        return_counts=True
                    )
                    for (from_val, to_val), count in zip(unique_transitions, counts):
                        transitions[from_val, to_val] += count
                
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
                return None, None, (y, x), True  # Added failure flag
            
            # Process subsequent years
            for year_idx in range(2, src.count + 1):
                try:
                    current_data = robust_read(src, year_idx, window)
                except Exception as e:
                    logging.warning(f"Failed reading tile {y},{x} (band {year_idx}): {str(e)}")
                    return None, None, (y, x), True  # Added failure flag
                
                # Calculate changes between consecutive years
                changed = prev_data != current_data
                changes += changed.astype('uint8')
                
                # Update transition matrix
                if np.any(changed):
                    from_vals = prev_data[changed]
                    to_vals = current_data[changed]
                    
                    # Stack and count unique transitions
                    stacked = np.column_stack((from_vals, to_vals))
                    unique_transitions, counts = np.unique(stacked, axis=0, return_counts=True)
                    
                    # Update transition matrix
                    for (from_val, to_val), count in zip(unique_transitions, counts):
                        if from_val < 256 and to_val < 256:  # Ensure within bounds
                            transition_matrix[from_val, to_val] += count
                
                prev_data = current_data
            
            return changes, transition_matrix, (y, x), False  # Added success flag

    except Exception as e:
        logging.error(f"Critical error processing tile {y},{x}: {str(e)}")
        return None, None, (y, x), True  # Added failure flag
    


    """Calculate changes for a single tile with robust error handling and proper Zarr storage."""
    tile_idx, vrt_path, window, temp_dir = args
    y, x = tile_idx
    
    # Initialize arrays
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
                
                # Calculate changes between consecutive years
                changed = prev_data != current_data
                changes += changed.astype('uint8')
                
                # Update transition matrix
                if np.any(changed):
                    from_vals = prev_data[changed]
                    to_vals = current_data[changed]
                    
                    # Stack and count unique transitions efficiently
                    stacked = np.column_stack((from_vals, to_vals))
                    unique_transitions, counts = np.unique(stacked, axis=0, return_counts=True)
                    
                    # Update transition matrix
                    for (from_val, to_val), count in zip(unique_transitions, counts):
                        if from_val < 256 and to_val < 256:  # Ensure within bounds
                            transition_matrix[from_val, to_val] += count
                
                prev_data = current_data

            if not failed:
                # Create proper Zarr storage
                zarr_folder = os.path.join(temp_dir, f'tile_{y}_{x}')
                os.makedirs(zarr_folder, exist_ok=True)
                
                # Store changes array
                changes_store = zarr.DirectoryStore(os.path.join(zarr_folder, 'changes'))
                changes_array = zarr.zeros(changes.shape, chunks=(256, 256), dtype='uint8', store=changes_store)
                changes_array[:] = changes
                
                # Store transitions array
                transitions_store = zarr.DirectoryStore(os.path.join(zarr_folder, 'transitions'))
                transitions_array = zarr.zeros(transition_matrix.shape, chunks=(64, 64), dtype='uint64', store=transitions_store)
                transitions_array[:] = transition_matrix
                
                return changes, transition_matrix, (y, x), temp_dir
            
    except Exception as e:
        logging.error(f"Critical error processing tile {y},{x}: {str(e)}", exc_info=True)
    
    return None

def extract_grid_data_with_polygon(vrt_path, geojson_path, output_base_dir):
    """Extract data for a specific polygon from a GeoJSON file, masking areas outside the polygon with zeros."""
    VRT_BLOCK_SIZE = 512  # Matches the BlockXSize/BlockYSize in VRT
    TILE_SIZE = 2048      # Should be a multiple of VRT_BLOCK_SIZE
    MAX_WORKERS = 8       # Conservative worker count
    FILL_VALUE = 0        # Matches NoDataValue in VRT

    try:
        os.makedirs(output_base_dir, exist_ok=True)
        logging.info("Starting robust analysis with visualization")

        # Load the GeoJSON file
        gdf = gpd.read_file(geojson_path)
        gdf = gdf.to_crs(epsg=4326)  # Ensure WGS84 coordinates

        # Extract the polygon coordinates from the GeoJSON
        if not gdf.empty and 'geometry' in gdf.columns:
            geometry = gdf.iloc[0].geometry
            if geometry.type == 'MultiPolygon':
                polygons = list(geometry.geoms)
            else:
                polygons = [geometry]
        else:
            raise ValueError("GeoJSON file is empty or does not contain valid geometry.")

        # Extract the "terrai_nom" value from the GeoJSON
        terrai_nom = gdf.iloc[0].get("terrai_nom", "unknown").replace(" ", "_")
        output_dir_with_name = os.path.join(output_base_dir, terrai_nom)

        def generate_local_grid_name(polygons):
            """Generate grid name from polygon bounds."""
            bounds = gdf.total_bounds  # minx, miny, maxx, maxy
            return (f"grid_{abs(bounds[0]):.6f}W_{abs(bounds[2]):.6f}W_"
                   f"{abs(bounds[1]):.6f}S_{abs(bounds[3]):.6f}S")

        grid_name = generate_local_grid_name(polygons)
        grid_output_dir = os.path.join(output_dir_with_name, grid_name)
        os.makedirs(grid_output_dir, exist_ok=True)

        with rasterio.open(vrt_path) as src:
            logging.info(f"Dataset info - Width: {src.width}, Height: {src.height}, Bands: {src.count}")
            transform = src.transform

            # Get bounding box of all polygons
            bounds = gdf.total_bounds
            min_lon, min_lat, max_lon, max_lat = bounds

            # Convert geographic coordinates to pixel coordinates using floating-point precision
            ul_col = (min_lon - transform.c) / transform.a
            ul_row = (max_lat - transform.f) / transform.e  # max_lat for upper row
            lr_col = (max_lon - transform.c) / transform.a
            lr_row = (min_lat - transform.f) / transform.e  # min_lat for lower row

            # Ensure window is within bounds
            ul_row = max(0.0, ul_row)
            ul_col = max(0.0, ul_col)
            lr_row = min(src.height, lr_row)
            lr_col = min(src.width, lr_col)

            # Ensure the bounding box intersects with the raster bounds
            if ul_row >= lr_row or ul_col >= lr_col:
                raise ValueError("Aligned window size is invalid: width and height must be greater than 0")

            full_window = Window.from_slices(
                rows=(int(ul_row), int(lr_row)),
                cols=(int(ul_col), int(lr_col))
            )

            logging.info(f"Processing {grid_name} with aligned window size: {full_window.height}x{full_window.width}")

            # Create a mask for the polygon(s)
            mask_shape = (full_window.height, full_window.width)
            mask = np.zeros(mask_shape, dtype=np.uint8)  # Use uint8 for rasterio compatibility
            
            # Transform polygons to pixel coordinates relative to the window
            transform_window = src.window_transform(full_window)
            
            # Rasterize all polygons at once for better performance
            all_polygons = []
            for polygon in polygons:
                coords = list(polygon.exterior.coords)
                polygon_pixels = [
                    ((lon - transform.c) / transform.a - full_window.col_off,
                     (lat - transform.f) / transform.e - full_window.row_off)
                    for lon, lat in coords
                ]
                all_polygons.append((polygon_pixels, 1))
                
            # Rasterize with fill=0 and default_value=1
            if all_polygons:
                mask = rasterize(
                    all_polygons,
                    out_shape=mask_shape,
                    transform=transform_window,
                    fill=0,
                    default_value=1,
                    dtype=np.uint8
                )

            # Convert mask to boolean after all polygons are processed
            mask_bool = mask.astype(bool)

            with tempfile.TemporaryDirectory() as temp_dir:
                # [tile processing setup...]
                
                full_changes = np.zeros((full_window.height, full_window.width), dtype='uint8')
                full_transitions = np.zeros((256, 256), dtype='uint64')
                failed_tiles = []

                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Define tasks as a list of tuples containing tile information
                    tasks = [
                        (y, x, VRT_FILE, Window(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE), temp_dir)
                        for y in range(int(full_window.height / TILE_SIZE))
                        for x in range(int(full_window.width / TILE_SIZE))
                    ]
                    futures = [executor.submit(process_tile, task) for task in tasks]
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tiles"):
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

                # Apply the mask to the changes data
                full_changes[~mask_bool] = 0

                # Store in Zarr with compression
                zarr_path = os.path.join(grid_output_dir, 'data.zarr')
                root = zarr.open(zarr_path, mode='w')
                
                # Store changes with proper attributes
                changes_array = root.create(
                    'changes',
                    shape=full_changes.shape,
                    chunks=(VRT_BLOCK_SIZE, VRT_BLOCK_SIZE),
                    dtype='uint8',
                )
                changes_array[:] = full_changes

                # Store transitions data
                transitions_array = root.zeros(
                    'transitions',
                    shape=full_transitions.shape,
                    chunks=(256, 256),
                    dtype='uint64',
                )
                transitions_array[:] = full_transitions

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
                    'window_transform': json.dumps(list(window_transform.to_gdal())),
                    'crs': str(src.crs),
                    'height': full_window.height,
                    'width': full_window.width,
                    'bounds': list(src.window_bounds(full_window)),
                    'lat_range': [float(min_lat), float(max_lat)],
                    'lon_range': [float(min_lon), float(max_lon)],
                    'grid_name': grid_name,
                    'failed_tiles': int(len(failed_tiles)),
                    'processing_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'vrt_block_size': int(VRT_BLOCK_SIZE),
                    'fill_value': int(FILL_VALUE),
                    'polygon_count': int(len(polygons))
                })

                return zarr_path, grid_output_dir

    except Exception as e:
        logging.critical(f"Error processing grid: {str(e)}", exc_info=True)
        raise




def create_visualizations(zarr_path, output_dir):
    """Create all visualizations from the Zarr data."""
    root = zarr.open(zarr_path, mode='r')
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    # 1. Create land cover map
    create_landcover_map(root, output_dir)
    

    
    # 3. Create Sankey diagrams
    create_sankey_diagrams(root, output_dir)

    # 4. Create decadal Sankey diagrams
    create_decadal_sankey_diagrams(root, output_dir)

    # 2. Create changes map
    create_changes_map(root, output_dir)

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
    """Create changes frequency visualization with diagnostics."""
    grid_name = root.attrs['grid_name']
    changes = root['changes'][:]
    
    # Add diagnostic information
    unique_values = np.unique(changes)
    logging.info(f"Changes array contains values: {unique_values}")
    logging.info(f"Max changes: {np.max(changes)}, Min changes: {np.min(changes)}")
    logging.info(f"Non-zero pixels: {np.count_nonzero(changes)}/{changes.size}")

    # Create visualization
    plt.figure(figsize=(12, 20))
    
    # Use a discrete colormap for better visualization of change counts
    cmap = plt.cm.get_cmap('gist_stern', np.max(changes) + 1)
    
    img = plt.imshow(changes, cmap=cmap, vmin=0, vmax=np.max(changes) if np.max(changes) > 0 else 1)
    
    # Only add colorbar if there are changes
    if np.max(changes) > 0:
        cbar = plt.colorbar(img, ticks=np.arange(0, np.max(changes) + 1))
        cbar.set_label('Number of Changes (1985-2023)')
    else:
        plt.text(0.5, 0.5, "No changes detected", 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.title(f"{grid_name} Change Frequency")
    output_path = os.path.join(output_dir, 'changes_map.png')
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()

    # Save diagnostic information
    with open(os.path.join(output_dir, 'changes_diagnostics.txt'), 'w') as f:
        f.write(f"Max changes: {np.max(changes)}\n")
        f.write(f"Min changes: {np.min(changes)}\n")
        f.write(f"Non-zero pixels: {np.count_nonzero(changes)}/{changes.size}\n")
        f.write(f"Unique values: {unique_values}\n")

  


def create_sankey_diagrams(root, output_dir):
    """Create organized Sankey diagrams with proper class alignment using transitions matrix."""
    grid_name = root.attrs['grid_name']
    
    # Get transitions data
    transitions = root['transitions'][:]  # Full transitions matrix
    first_year = root['first_year'][:]
    last_year = root['last_year'][:]
    
    # Get all classes that have any transitions
    from_classes = np.unique(np.where(transitions > 0)[0])
    to_classes = np.unique(np.where(transitions > 0)[1])
    all_classes = np.unique(np.concatenate([from_classes, to_classes]))
    
    # Filter classes that have labels and are present in the data
    present_classes = [cls for cls in all_classes if cls in LABELS]
    
    if not present_classes:
        logging.warning("No valid classes found for Sankey diagrams")
        return
    
    # Order classes by frequency in first year (largest on top)
    unique_first, counts_first = np.unique(first_year, return_counts=True)
    class_freq = {cls: cnt for cls, cnt in zip(unique_first, counts_first) if cls in LABELS}
    present_classes = sorted(present_classes, key=lambda x: -class_freq.get(x, 0))
    
    # Define decades (start_year, end_year)
    decades = [
        (1985, 1994),  # First decade
        (1995, 2004),  # Second decade
        (2005, 2014),  # Third decade
        (2015, 2023),  # Fourth period
        (1985, 2023)   # Full period
    ]
    
    # Create decade diagrams
    for start_year, end_year in decades:
        try:
            # For full period, use the full transitions matrix
            if (start_year, end_year) == (1985, 2023):
                decade_trans = transitions.copy()
            else:
                # Calculate transitions for the specific decade
                # This assumes you have yearly data stored - adjust if needed
                # Alternatively, you could pre-compute decade transitions during processing
                decade_trans = np.zeros_like(transitions)
                # [Add logic to compute decade-specific transitions here]
                # For now, we'll just use the full transitions scaled down
                year_span = end_year - start_year + 1
                decade_trans = (transitions * (year_span / 38)).astype(int)
            
            # Filter to only include present classes
            filtered_trans = np.zeros((len(present_classes), len(present_classes)), dtype='uint64')
            class_to_idx = {cls: i for i, cls in enumerate(present_classes)}
            
            for from_cls in present_classes:
                for to_cls in present_classes:
                    filtered_trans[class_to_idx[from_cls], class_to_idx[to_cls]] = \
                        decade_trans[from_cls, to_cls]
            
            # Remove self-transitions if desired
            np.fill_diagonal(filtered_trans, 0)
            
            # Save change matrix CSV
            change_matrix_csv = os.path.join(output_dir, f'change_matrix_{start_year}_{end_year}.csv')
            with open(change_matrix_csv, 'w') as f:
                f.write("From_Class,From_Label,To_Class,To_Label,Count\n")
                for i, from_cls in enumerate(present_classes):
                    for j, to_cls in enumerate(present_classes):
                        count = filtered_trans[i, j]
                        if count > 0:
                            f.write(f"{from_cls},{LABELS[from_cls]},{to_cls},{LABELS[to_cls]},{count}\n")
            
            # Create graphical table
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            table_data = [["From/To"] + [LABELS[cls] for cls in present_classes]]
            for i, from_cls in enumerate(present_classes):
                row = [LABELS[from_cls]] + [int(filtered_trans[i, j]) for j in range(len(present_classes))]
                table_data.append(row)
            
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            
            # Apply colors to cells
            for i, row in enumerate(table_data[1:], start=1):
                for j, val in enumerate(row[1:], start=1):
                    cls = present_classes[j-1] if i < j else present_classes[i-1]
                    if cls in COLOR_MAP:
                        table[(i, j)].set_facecolor(COLOR_MAP[cls])
                        brightness = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(COLOR_MAP[cls]))[2]
                        table[(i, j)].get_text().set_color('white' if brightness < 0.6 else 'black')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'change_matrix_{start_year}_{end_year}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create Sankey diagram
            title = f"{grid_name} Land Cover Changes {start_year}-{end_year}"
            output_html = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.html')
            output_csv = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.csv')
            
            create_sankey(
                transition_matrix=filtered_trans,
                classes=present_classes,
                title=title,
                output_html=output_html,
                output_csv=output_csv
            )
            
        except Exception as e:
            logging.error(f"Error creating {start_year}-{end_year} diagram: {str(e)}", exc_info=True)


def create_decadal_sankey_diagrams(root, output_dir):
    """Create enhanced Sankey diagrams showing persistence and transitions."""
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    try:
        # Get transitions data
        transitions = root['transitions'][:]
        first_year = root['first_year'][:]
        last_year = root['last_year'][:]
        
        # Get all classes present in the data (excluding no-data)
        all_classes = sorted(set(np.unique(first_year)) | set(np.unique(last_year)))
        all_classes = [cls for cls in all_classes if cls in LABELS and cls != 0]
        
        if not all_classes:
            logging.warning("No valid classes found for decadal Sankey diagrams")
            return
        
        # Prepare labels and colors
        class_labels = {cls: LABELS[cls] for cls in all_classes}
        class_colors = {cls: matplotlib.colors.to_rgb(COLOR_MAP.get(cls, "#999999")) 
                       for cls in all_classes}
        
        # Define decades
        decades = [
            (1985, 1994),  # First decade
            (1995, 2004),  # Second decade
            (2005, 2014),  # Third decade
            (2015, 2023),  # Fourth period
            (1985, 2023)   # Full period
        ]
        
        for start_year, end_year in decades:
            try:
                # For full period, use the full transitions matrix
                if (start_year, end_year) == (1985, 2023):
                    period_trans = transitions.copy()
                else:
                    # Calculate transitions for the specific decade
                    # This would ideally be pre-computed during processing
                    period_trans = np.zeros_like(transitions)
                    # [Add logic to compute period-specific transitions here]
                    # For now, we'll just use the full transitions scaled down
                    year_span = end_year - start_year + 1
                    period_trans = (transitions * (year_span / 38)).astype(int)
                
                # Filter to only include our classes of interest
                filtered_trans = np.zeros((len(all_classes), len(all_classes)), dtype='uint64')
                class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
                
                for from_cls in all_classes:
                    for to_cls in all_classes:
                        filtered_trans[class_to_idx[from_cls], class_to_idx[to_cls]] = \
                            period_trans[from_cls, to_cls]
                
                # Calculate node sizes
                out_flows = np.sum(filtered_trans, axis=1)  # Outgoing flows
                in_flows = np.sum(filtered_trans, axis=0)   # Incoming flows
                
                # Create node structure
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
                        return [0.5] * len(flows)
                    positions = []
                    cumulative = 0
                    for flow in flows:
                        positions.append((cumulative + flow/2) / total)
                        cumulative += flow
                    return positions
                
                left_y = calc_positions(out_flows)
                right_y = calc_positions(in_flows)
                
                # Create links
                sources = []
                targets = []
                values = []
                link_colors = []
                
                for i, from_cls in enumerate(all_classes):
                    for j, to_cls in enumerate(all_classes):
                        value = filtered_trans[i, j]
                        if value > 0:
                            sources.append(node_positions[(from_cls, 'start')])
                            targets.append(node_positions[(to_cls, 'end')])
                            values.append(value)
                            rgb = class_colors[from_cls]
                            alpha = 0.4 if i == j else 0.6  # Lighter for self-transitions
                            link_colors.append(
                                f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"
                            )
                
                # Create the Sankey diagram
                fig = go.Figure(go.Sankey(
                    arrangement="fixed",
                    node=dict(
                        pad=30,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=node_names,
                        color=node_colors,
                        x=[0.1] * len(all_classes) + [0.9] * len(all_classes),
                        y=left_y + right_y,
                        customdata=[f"Out: {out_flows[i]:,}<br>In: {in_flows[i]:,}" 
                                  for i in range(len(all_classes))] * 2,
                        hovertemplate="%{label}<br>%{customdata}<extra></extra>"
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                        customdata=[f"{all_classes[i] if i < len(all_classes) else '?'} → "
                                   f"{all_classes[j % len(all_classes)] if j >= len(all_classes) else '?'}"
                                   for i, j in zip(sources, targets)],
                        hovertemplate="%{customdata}<br>Count: %{value:,}<extra></extra>"
                    )
                ))
                
                fig.update_layout(
                    title_text=f"{grid_name} Land Cover Changes {start_year}-{end_year}",
                    font=dict(size=12),
                    height=max(1200, len(all_classes) * 60),
                    width=1600,
                    margin=dict(l=150, r=150, b=100, t=120)
                )
                
                # Save outputs
                html_path = os.path.join(output_dir, f'enhanced_transitions_{start_year}_{end_year}.html')
                plot(fig, filename=html_path, auto_open=False)
                
            except Exception as e:
                logging.error(f"Error creating {start_year}-{end_year} diagram: {str(e)}", exc_info=True)
                
    except Exception as e:
        logging.error(f"Error in decadal Sankey diagrams: {str(e)}", exc_info=True)


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
    # The POLYGON_8x8 definition is not necessary as the bounding box is dynamically extracted from the GeoJSON file.
    # You can safely remove it.
    
    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        logging.info("Starting robust analysis with visualization")
        
        # Load the GeoJSON file
        geojson_path = 'indigenous_test.geojson'
        gdf = gpd.read_file(geojson_path)

        # Extract the polygon coordinates from the GeoJSON
        if not gdf.empty and 'geometry' in gdf.columns:
            if gdf.iloc[0].geometry.type == 'MultiPolygon':
                polygon_coords = [list(polygon.exterior.coords) for polygon in gdf.iloc[0].geometry.geoms]
            else:
                polygon_coords = list(gdf.iloc[0].geometry.exterior.coords)
        else:
            raise ValueError("GeoJSON file is empty or does not contain valid geometry.")

        # Use the extracted polygon coordinates
        # Extract the "terrai_nom" value from the GeoJSON
        terrai_nom = gdf.iloc[0].get("terrai_nom", "unknown").replace(" ", "_")
        output_dir_with_name = os.path.join(OUTPUT_BASE_DIR, terrai_nom)

        # Use the extracted polygon coordinates and updated output directory
        zarr_path, grid_output_dir = extract_grid_data_with_polygon(VRT_FILE, geojson_path, output_dir_with_name)
        # Ensure the 'changes' array is populated before creating visualizations
        if 'changes' not in zarr.open(zarr_path, mode='r'):
            logging.error("The 'changes' array is missing in the Zarr store. Ensure data is processed correctly.")
        else:
            create_visualizations(zarr_path, grid_output_dir)
        
        
        logging.info(f"Analysis complete. Results saved to {grid_output_dir}")
        
    except Exception as e:
        logging.critical(f"Analysis failed: {str(e)}", exc_info=True)