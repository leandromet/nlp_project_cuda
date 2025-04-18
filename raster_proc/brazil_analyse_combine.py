import zarr
import numpy as np
import os
import json
from tqdm import tqdm
import logging
import plotly.graph_objects as go
from plotly.offline import plot
import matplotlib.colors
import matplotlib.pyplot as plt
from rasterio.transform import Affine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('combined_analysis.log'),
        logging.StreamHandler()
    ]
)

# Color mapping and labels (same as before)
COLOR_MAP = {...}  # Keep your existing color map
LABELS = {...}     # Keep your existing labels

def combine_zarr_datasets(input_base_dir, output_dir, max_chunk_size=512):
    """Combine all individual grid zarr datasets into one comprehensive dataset using chunked processing."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        grid_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
        
        if not grid_dirs:
            raise ValueError(f"No grid directories found in {input_base_dir}")
        
        # First pass: collect metadata and calculate total dimensions
        logging.info("Collecting metadata from all grids...")
        total_height = 0
        total_width = 0
        num_years = None
        bounds = None
        window_transforms = []
        
        for grid_dir in tqdm(grid_dirs):
            zarr_path = os.path.join(input_base_dir, grid_dir, 'data.zarr')
            if not os.path.exists(zarr_path):
                logging.warning(f"Zarr file not found in {grid_dir}, skipping")
                continue
            
            root = zarr.open(zarr_path, mode='r')
            if num_years is None:
                num_years = root['data'].shape[0]
            total_height += root['data'].shape[1]
            total_width = max(total_width, root['data'].shape[2])
            
            # Collect bounds and transforms
            window_transforms.append(Affine(*json.loads(root.attrs['window_transform'])))
            if bounds is None:
                bounds = list(root.attrs['bounds'])
            else:
                bounds[0] = min(bounds[0], root.attrs['bounds'][0])
                bounds[1] = min(bounds[1], root.attrs['bounds'][1])
                bounds[2] = max(bounds[2], root.attrs['bounds'][2])
                bounds[3] = max(bounds[3], root.attrs['bounds'][3])
        
        if num_years is None:
            raise ValueError("No valid zarr datasets found to combine")
        
        # Create combined zarr store with chunked storage
        combined_zarr_path = os.path.join(output_dir, 'combined_data.zarr')
        root = zarr.open(combined_zarr_path, mode='w')
        
        # Initialize arrays with chunking
        data_array = root.zeros(
            'data',
            shape=(num_years, total_height, total_width),
            chunks=(1, max_chunk_size, max_chunk_size),
            dtype='uint8'
        )
        
        changes_array = root.zeros(
            'changes',
            shape=(total_height, total_width),
            chunks=(max_chunk_size, max_chunk_size),
            dtype='uint8'
        )
        
        transitions_array = root.zeros(
            'transitions',
            shape=(256, 256),
            dtype='uint64'
        )
        
        # Second pass: combine the data in chunks
        logging.info("Combining data from all grids...")
        current_row = 0
        for grid_dir in tqdm(grid_dirs):
            zarr_path = os.path.join(input_base_dir, grid_dir, 'data.zarr')
            if not os.path.exists(zarr_path):
                continue
            
            root = zarr.open(zarr_path, mode='r')
            grid_data = root['data']
            grid_height = grid_data.shape[1]
            grid_width = grid_data.shape[2]
            
            # Process each year's data
            for year_idx in range(num_years):
                data_array[year_idx, current_row:current_row+grid_height, :grid_width] = grid_data[year_idx]
            
            # Process changes
            changes_array[current_row:current_row+grid_height, :grid_width] += root['changes'][:]
            
            # Process transitions
            transitions_array[:] += root['transitions'][:]
            
            current_row += grid_height
        
        # Store metadata
        root.attrs.update({
            'window_transform': json.dumps(window_transforms[0].to_gdal()),
            'crs': 'EPSG:4326',
            'height': total_height,
            'width': total_width,
            'bounds': bounds,
            'source_grids': grid_dirs
        })
        
        logging.info(f"Combined dataset created at {combined_zarr_path}")
        return combined_zarr_path
        
    except Exception as e:
        logging.error(f"Error combining datasets: {str(e)}", exc_info=True)
        raise

def create_full_sankey_diagrams(zarr_path, output_dir, sample_fraction=0.01):
    """Create Sankey diagrams for the full combined dataset using sampling."""
    try:
        root = zarr.open(zarr_path, mode='r')
        years = list(range(1985, 2024))
        decadal_windows = [(1985, 1995), (1995, 2005), (2005, 2015), (2015, 2023)]
        
        # Get all possible classes from the data
        all_possible_classes = sorted(set(np.unique(root['data'][:])))
        class_labels = {cls: LABELS.get(cls, f"Class {cls}") for cls in all_possible_classes}
        class_colors = {cls: matplotlib.colors.to_rgb(COLOR_MAP.get(cls, "#999999")) 
                       for cls in all_possible_classes}

        for start_year, end_year in decadal_windows:
            try:
                start_idx = years.index(start_year)
                end_idx = years.index(end_year)
                
                # Sample data to reduce memory usage
                sample_size = int(root['data'].shape[1] * root['data'].shape[2] * sample_fraction)
                random_indices = np.random.choice(root['data'].shape[1] * root['data'].shape[2], 
                                                 size=sample_size, replace=False)
                
                # Convert flat indices to 2D indices
                rows = random_indices // root['data'].shape[2]
                cols = random_indices % root['data'].shape[2]
                
                # Get sampled data
                start_data = np.array([root['data'][start_idx, r, c] for r, c in zip(rows, cols)])
                end_data = np.array([root['data'][end_idx, r, c] for r, c in zip(rows, cols)])
                
                # Calculate persistence and transitions from sampled data
                persistence_mask = (start_data == end_data)
                persisted_classes, persisted_counts = np.unique(start_data[persistence_mask], 
                                                              return_counts=True)
                
                transition_mask = (start_data != end_data)
                transition_pairs, transition_counts = np.unique(
                    np.vstack((start_data[transition_mask], end_data[transition_mask])).T,
                    axis=0,
                    return_counts=True
                )
                
                # Scale counts back to full dataset size
                scale_factor = 1 / sample_fraction
                persisted_counts = (persisted_counts * scale_factor).astype(int)
                transition_counts = (transition_counts * scale_factor).astype(int)
                
                # Filter small transitions (<0.1% of total changes)
                min_transitions = np.sum(transition_counts) * 0.001
                significant_transitions = transition_counts > min_transitions
                transition_pairs = transition_pairs[significant_transitions]
                transition_counts = transition_counts[significant_transitions]
                
                # Create Sankey diagram (same as before)
                # ... [rest of the Sankey diagram code remains the same]
                
                # Save files with "sampled" in name to indicate methodology
                html_path = os.path.join(output_dir, f'sampled_transitions_{start_year}_{end_year}.html')
                png_path = os.path.join(output_dir, f'sampled_transitions_{start_year}_{end_year}.png')
                
                plot(fig, filename=html_path, auto_open=False, include_plotlyjs='cdn')
                
                try:
                    fig.write_image(png_path, scale=2, engine="kaleido")
                    logging.info(f"Saved sampled diagram to {png_path}")
                except Exception as e:
                    logging.warning(f"Could not save static image: {str(e)}")
                
                logging.info(f"Created sampled diagram for {start_year}-{end_year}")
                
            except Exception as e:
                logging.error(f"Error creating {start_year}-{end_year} diagram: {str(e)}", exc_info=True)
                
    except Exception as e:
        logging.error(f"Error in full dataset Sankey diagrams: {str(e)}", exc_info=True)
        raise

def visualize_full_results(zarr_path, output_dir, downsample_factor=10):
    """Create visualizations for the full combined dataset with downsampling."""
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Get stored metadata
        window_transform = Affine(*json.loads(root.attrs['window_transform']))
        bounds = root.attrs['bounds']
        
        # Downsample data for visualization
        data = root['data'][-1, ::downsample_factor, ::downsample_factor]
        changes = root['changes'][::downsample_factor, ::downsample_factor]
        
        # Create visualizations (same as before but with downsampled data)
        # ... [rest of the visualization code remains the same]
        
        plt.savefig(
            os.path.join(output_dir, f'downsampled_{downsample_factor}_full_dataset_2023.png'),
            dpi=300,
            bbox_inches='tight'
        )
        
        logging.info("Full dataset visualization complete (downsampled)")
        
    except Exception as e:
        logging.error(f"Error visualizing full dataset results: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    INPUT_BASE_DIR = '/srv/extrassd/mapbiomas_proc_zarr'
    OUTPUT_DIR = '/media/bndt-ai/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/combined_results'
    
    try:
        logging.info("Starting memory-efficient combined analysis")
        
        # Combine all individual zarr datasets with chunked processing
        combined_zarr_path = combine_zarr_datasets(INPUT_BASE_DIR, OUTPUT_DIR)
        
        # Create visualizations with downsampling
        visualize_full_results(combined_zarr_path, OUTPUT_DIR, downsample_factor=10)
        
        # Create Sankey diagrams with sampling
        create_full_sankey_diagrams(combined_zarr_path, OUTPUT_DIR, sample_fraction=0.01)
        
        logging.info(f"Combined analysis complete. Results saved to {OUTPUT_DIR}")
        
    except Exception as e:
        logging.critical(f"Combined analysis failed: {str(e)}", exc_info=True)

