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
import cupy as cp  # Replace numpy for GPU operations
import concurrent
from concurrent.futures import ProcessPoolExecutor


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


def visualize_full_results(zarr_path, output_dir, downsample_factor=20):
    """Generate land cover and change frequency maps for Brazil"""
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Brazil geographic bounds (approximate)
        brazil_bounds = {
            'min_lon': -74.0, 'max_lon': -34.0,
            'min_lat': -33.7, 'max_lat': 5.2
        }
        
        # Get data transform
        transform = Affine(*json.loads(root.attrs.get('window_transform', '[1,0,0,0,1,0]')))
        
        # Convert geo to pixel coordinates
        def geo_to_pixel(lon, lat):
            col = int((lon - transform.c) / transform.a)
            row = int((lat - transform.f) / transform.e)
            return max(0, row), max(0, col)
        
        # Get Brazil's pixel bounds
        row_start, col_start = geo_to_pixel(brazil_bounds['min_lon'], brazil_bounds['max_lat'])
        row_end, col_end = geo_to_pixel(brazil_bounds['max_lon'], brazil_bounds['min_lat'])
        
        # Extract data with downsampling
        data = root['data'][-1, row_start:row_end:downsample_factor, 
                           col_start:col_end:downsample_factor]
        changes = root['changes'][row_start:row_end:downsample_factor, 
                                col_start:col_end:downsample_factor]
        
        # Create land cover map
        plt.figure(figsize=(20, 15))
        unique_vals = sorted(COLOR_MAP.keys())
        cmap = ListedColormap([COLOR_MAP[v] for v in unique_vals])
        norm = BoundaryNorm([v-0.5 for v in unique_vals] + [unique_vals[-1]+0.5], len(unique_vals))
        
        plt.imshow(data, cmap=cmap, norm=norm,
                 extent=[brazil_bounds['min_lon'], brazil_bounds['max_lon'],
                         brazil_bounds['min_lat'], brazil_bounds['max_lat']],
                 aspect='auto')
        plt.title('Brazil Land Cover 2023')
        plt.savefig(os.path.join(output_dir, 'brazil_landcover.png'), dpi=300)
        plt.close()
        
        # Create change frequency map
        plt.figure(figsize=(20, 15))
        plt.imshow(changes, cmap='viridis',
                 extent=[brazil_bounds['min_lon'], brazil_bounds['max_lon'],
                         brazil_bounds['min_lat'], brazil_bounds['max_lat']],
                 aspect='auto')
        plt.title('Brazil Land Cover Change Frequency (1985-2023)')
        plt.colorbar(label='Number of Changes')
        plt.savefig(os.path.join(output_dir, 'brazil_changes.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")

def create_full_sankey_diagrams(zarr_path, output_dir, sample_size=1000000):
    """Generate Sankey diagrams for land cover transitions"""
    try:
        root = zarr.open(zarr_path, mode='r')
        years = list(range(1985, 2024))
        decadal_windows = [(1985,1995), (1995,2005), (2005,2015), (2015,2023)]
        
        # Get all classes present in data
        sample = root['data'][:, ::100, ::100]  # Coarse sample
        all_classes = sorted(np.unique(sample))
        
        for start_year, end_year in decadal_windows:
            try:
                # Get random sample
                idx = np.random.choice(root['data'].shape[1] * root['data'].shape[2], 
                                     size=sample_size, replace=False)
                rows, cols = np.unravel_index(idx, (root['data'].shape[1], root['data'].shape[2]))
                
                # Get data
                start_data = root['data'][years.index(start_year), rows, cols]
                end_data = root['data'][years.index(end_year), rows, cols]
                
                # Calculate transitions
                changed = start_data != end_data
                transition_pairs, transition_counts = np.unique(
                    np.vstack((start_data[changed], end_data[changed])).T,
                    axis=0, return_counts=True
                )
                
                # Calculate persistence
                same = start_data == end_data
                persisted_classes, persisted_counts = np.unique(start_data[same], return_counts=True)
                
                # Create Sankey diagram
                fig = go.Figure(go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        label=[LABELS.get(c, f"Class {c}") for c in all_classes] * 2,
                        color=[COLOR_MAP.get(c, "#999999") for c in all_classes] * 2
                    ),
                    link=dict(
                        source=[all_classes.index(p[0]) for p in transition_pairs],
                        target=[len(all_classes) + all_classes.index(p[1]) for p in transition_pairs],
                        value=transition_counts,
                        color=[f"{COLOR_MAP.get(p[0], '#999999')}88" for p in transition_pairs]
                    )
                ))
                
                fig.update_layout(title_text=f"Brazil Land Cover Changes {start_year}-{end_year}")
                
                # Save outputs
                fig.write_html(os.path.join(output_dir, f'sankey_{start_year}_{end_year}.html'))
                fig.write_image(os.path.join(output_dir, f'sankey_{start_year}_{end_year}.png'))
                
            except Exception as e:
                logging.error(f"Error in {start_year}-{end_year}: {str(e)}")
                
    except Exception as e:
        logging.error(f"Sankey generation failed: {str(e)}")

if __name__ == '__main__':
    OUTPUT_DIR = '/media/bndt-ai/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/combined_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, 'analysis.log')),
            logging.StreamHandler()
        ]
    )
    
    try:
        zarr_path = os.path.join(OUTPUT_DIR, 'combined_data.zarr')
        
        # Generate maps
        visualize_full_results(zarr_path, OUTPUT_DIR)
        
        # Generate Sankey diagrams
        create_full_sankey_diagrams(zarr_path, OUTPUT_DIR)
        
    except Exception as e:
        logging.critical(f"Analysis failed: {str(e)}")



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

def process_decade(zarr_path, start_year, end_year, output_dir):
    """GPU-accelerated decade processing"""
    try:
        # Define the range of years
        years = list(range(1985, 2024))
        
        # Load data to GPU (chunked)
        root = zarr.open(zarr_path)
        start_data = cp.asarray(root['data'][years.index(start_year), ::10, ::10])
        end_data = cp.asarray(root['data'][years.index(end_year), ::10, ::10])
        
        # GPU-optimized transition matrix
        changes = start_data != end_data
        transitions = cp.zeros((256, 256), dtype='uint64')
        
        # [Add GPU-based transition calculation]
        
        # Save results
        cp.save(os.path.join(output_dir, f'transitions_{start_year}_{end_year}.npy'), 
               cp.asnumpy(transitions))
        
    except Exception as e:
        logging.error(f"Decade {start_year}-{end_year} failed: {str(e)}")


if __name__ == '__main__':
    # Config
    CPU_WORKERS = 10  # Half your cores
    OUTPUT_DIR = '/media/bndt-ai/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/combined_results'
    
    # Configure logging first
    configure_logging(OUTPUT_DIR)
    
    try:
        # Verify Zarr file exists
        combined_zarr = os.path.join(OUTPUT_DIR, 'combined_data.zarr')
        if not os.path.exists(combined_zarr):
            raise FileNotFoundError(f"Zarr file not found at {combined_zarr}")
        
        # Increase system limits
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
        
        # Run visualization (single process)
        logging.info("Starting visualization...")
        visualize_full_results(combined_zarr, OUTPUT_DIR)
        
        # Process decades in parallel
        logging.info("Starting parallel decade processing...")
        DECADES = [(1985, 1995), (1995, 2005), (2005, 2015), (2015, 2023)]
        
        with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
            futures = []
            for start, end in DECADES:
                futures.append(executor.submit(
                    process_decade,
                    zarr_path=combined_zarr,
                    start_year=start,
                    end_year=end,
                    output_dir=OUTPUT_DIR
                ))
            
            # Monitor progress
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # This will raise exceptions if any occurred
                    logging.info(f"Completed processing {future}")
                except Exception as e:
                    logging.error(f"Error in parallel processing: {str(e)}")
        
        # Generate Sankey diagrams after all decades are processed
        logging.info("Starting Sankey diagram generation...")
        create_full_sankey_diagrams(combined_zarr, OUTPUT_DIR)
        
    except Exception as e:
        logging.critical(f"Main execution failed: {str(e)}")
    finally:
        logging.info("===== PROCESS COMPLETED =====")
        sys.stdout.close()
        sys.stderr.close()