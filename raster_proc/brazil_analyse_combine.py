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

# Color mapping and labels (same as in your original script)
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

def combine_zarr_datasets(input_base_dir, output_dir):
    """Combine all individual grid zarr datasets into one comprehensive dataset."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        grid_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
        
        if not grid_dirs:
            raise ValueError(f"No grid directories found in {input_base_dir}")
        
        # Initialize variables to store combined data
        combined_data = None
        combined_changes = None
        combined_transitions = np.zeros((256, 256), dtype='uint64')
        total_height = 0
        total_width = 0
        bounds = None
        window_transforms = []
        
        # First pass: collect metadata and calculate total dimensions
        logging.info("Collecting metadata from all grids...")
        for grid_dir in tqdm(grid_dirs):
            zarr_path = os.path.join(input_base_dir, grid_dir, 'data.zarr')
            if not os.path.exists(zarr_path):
                logging.warning(f"Zarr file not found in {grid_dir}, skipping")
                continue
            
            root = zarr.open(zarr_path, mode='r')
            if combined_data is None:
                # Initialize combined arrays based on first grid
                combined_data = np.zeros((root['data'].shape[0], 0, 0), dtype=root['data'].dtype)
                combined_changes = np.zeros((0, 0), dtype=root['changes'].dtype)
            
            total_height += root['data'].shape[1]
            total_width = max(total_width, root['data'].shape[2])  # Assuming all grids have same width
            
            # Collect bounds and transforms
            window_transforms.append(Affine(*json.loads(root.attrs['window_transform'])))
            if bounds is None:
                bounds = list(root.attrs['bounds'])
            else:
                # Expand bounds to encompass all grids
                bounds[0] = min(bounds[0], root.attrs['bounds'][0])  # min west
                bounds[1] = min(bounds[1], root.attrs['bounds'][1])  # min south
                bounds[2] = max(bounds[2], root.attrs['bounds'][2])  # max east
                bounds[3] = max(bounds[3], root.attrs['bounds'][3])  # max north
        
        if combined_data is None:
            raise ValueError("No valid zarr datasets found to combine")
        
        # Resize combined arrays
        combined_data = np.zeros((combined_data.shape[0], total_height, total_width), dtype=combined_data.dtype)
        combined_changes = np.zeros((total_height, total_width), dtype=combined_changes.dtype)
        
        # Second pass: combine the data
        logging.info("Combining data from all grids...")
        current_row = 0
        for grid_dir in tqdm(grid_dirs):
            zarr_path = os.path.join(input_base_dir, grid_dir, 'data.zarr')
            if not os.path.exists(zarr_path):
                continue
            
            root = zarr.open(zarr_path, mode='r')
            grid_data = root['data'][:]
            grid_height = grid_data.shape[1]
            
            # Combine data
            combined_data[:, current_row:current_row+grid_height, :grid_data.shape[2]] = grid_data
            
            # Combine changes
            combined_changes[current_row:current_row+grid_height, :root['changes'].shape[1]] += root['changes'][:]
            
            # Combine transitions
            combined_transitions += root['transitions'][:]
            
            current_row += grid_height
        
        # Create combined zarr store
        combined_zarr_path = os.path.join(output_dir, 'combined_data.zarr')
        root = zarr.open(combined_zarr_path, mode='w')
        
        # Store combined data
        root.zeros('data', shape=combined_data.shape, chunks=(1, 512, 512), dtype=combined_data.dtype)[:] = combined_data
        root.zeros('changes', shape=combined_changes.shape, chunks=(512, 512), dtype=combined_changes.dtype)[:] = combined_changes
        root.zeros('transitions', shape=combined_transitions.shape, dtype=combined_transitions.dtype)[:] = combined_transitions
        
        # Store metadata
        root.attrs.update({
            'window_transform': json.dumps(window_transforms[0].to_gdal()),  # Using first transform as reference
            'crs': root.attrs.get('crs', 'EPSG:4326'),
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

def create_full_sankey_diagrams(zarr_path, output_dir):
    """Create Sankey diagrams for the full combined dataset."""
    try:
        root = zarr.open(zarr_path, mode='r')
        data = root['data'][:]
        years = list(range(1985, 2024))
        decadal_windows = [(1985, 1995), (1995, 2005), (2005, 2015), (2015, 2023)]
        
        # Get all possible classes from the data
        all_possible_classes = sorted(set(data.flatten()))
        class_labels = {cls: LABELS.get(cls, f"Class {cls}") for cls in all_possible_classes}
        class_colors = {cls: matplotlib.colors.to_rgb(COLOR_MAP.get(cls, "#999999")) 
                       for cls in all_possible_classes}

        for start_year, end_year in decadal_windows:
            try:
                start_idx = years.index(start_year)
                end_idx = years.index(end_year)
                
                start_data = data[start_idx]
                end_data = data[end_idx]
                
                # Calculate persistence (unchanged pixels)
                persistence_mask = (start_data == end_data)
                persisted_classes, persisted_counts = np.unique(start_data[persistence_mask], 
                                                              return_counts=True)
                
                # Calculate transitions (changed pixels)
                transition_mask = (start_data != end_data)
                transition_pairs, transition_counts = np.unique(
                    np.vstack((start_data[transition_mask], end_data[transition_mask])).T,
                    axis=0,
                    return_counts=True
                )
                
                # Filter small transitions (<0.1% of total changes)
                min_transitions = np.sum(transition_counts) * 0.001
                significant_transitions = transition_counts > min_transitions
                transition_pairs = transition_pairs[significant_transitions]
                transition_counts = transition_counts[significant_transitions]
                
                # Combine all classes that appear in either persisted or transitioned data
                all_classes = sorted(set(persisted_classes) | 
                                   set(transition_pairs[:,0]) | 
                                   set(transition_pairs[:,1]))
                
                # Create node structure - duplicate nodes for left/right sides
                node_names = []
                node_colors = []
                node_positions = {}
                
                # Left nodes (start year)
                for i, cls in enumerate(all_classes):
                    label = f"{cls}: {class_labels[cls]} (Start)"
                    node_names.append(label)
                    node_positions[(cls, 'start')] = i
                    rgb = class_colors[cls]
                    node_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)")
                
                # Right nodes (end year)
                right_offset = len(all_classes)
                for i, cls in enumerate(all_classes):
                    label = f"{cls}: {class_labels[cls]} (End)"
                    node_names.append(label)
                    node_positions[(cls, 'end')] = right_offset + i
                    rgb = class_colors[cls]
                    node_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)")
                
                # Create links
                sources = []
                targets = []
                values = []
                link_colors = []
                
                # Add persistence flows
                for cls, count in zip(persisted_classes, persisted_counts):
                    sources.append(node_positions[(cls, 'start')])
                    targets.append(node_positions[(cls, 'end')])
                    values.append(int(count))
                    rgb = class_colors[cls]
                    link_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.4)")
                
                # Add transition flows
                for (from_cls, to_cls), count in zip(transition_pairs, transition_counts):
                    sources.append(node_positions[(from_cls, 'start')])
                    targets.append(node_positions[(to_cls, 'end')])
                    values.append(int(count))
                    rgb = class_colors[from_cls]
                    link_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.6)")
                
                # Create the Sankey diagram
                fig = go.Figure(go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=30,
                        thickness=25,
                        line=dict(color="black", width=0.5),
                        label=node_names,
                        color=node_colors,
                        x=[0.1] * len(all_classes) + [0.9] * len(all_classes),  # Left/right positioning
                        y=[i/(len(all_classes)+1) for i in range(len(all_classes))] * 2,  # Even spacing
                        hoverinfo='all'
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                        hoverinfo='all'
                    )
                ))
                
                fig.update_layout(
                    title_text=f"Full Dataset Land Cover Changes {start_year}-{end_year}<br>"
                             f"(Showing persistence and transitions)",
                    font=dict(size=12, family="Arial"),
                    height=1200,
                    width=1600,
                    margin=dict(l=100, r=100, b=100, t=120, pad=20)
                )
                
                # Save files
                html_path = os.path.join(output_dir, f'full_transitions_{start_year}_{end_year}.html')
                png_path = os.path.join(output_dir, f'full_transitions_{start_year}_{end_year}.png')
                
                plot(fig, filename=html_path, auto_open=False, include_plotlyjs='cdn')
                
                try:
                    fig.write_image(png_path, scale=2, engine="kaleido")
                    logging.info(f"Saved diagram to {png_path}")
                except Exception as e:
                    logging.warning(f"Could not save static image: {str(e)}")
                
                logging.info(f"Created full dataset diagram for {start_year}-{end_year}")
                
            except Exception as e:
                logging.error(f"Error creating {start_year}-{end_year} diagram: {str(e)}", exc_info=True)
                
    except Exception as e:
        logging.error(f"Error in full dataset Sankey diagrams: {str(e)}", exc_info=True)
        raise

def visualize_full_results(zarr_path, output_dir):
    """Create visualizations for the full combined dataset."""
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Get stored metadata
        window_transform = Affine(*json.loads(root.attrs['window_transform']))
        bounds = root.attrs['bounds']
        
        # Create figure for land cover with proper geographic coordinates
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create colormap
        unique_values = sorted(COLOR_MAP.keys())
        colors = [COLOR_MAP[val] for val in unique_values]
        cmap = ListedColormap(colors)
        
        # Show land cover data
        from matplotlib.colors import BoundaryNorm
        norm = BoundaryNorm(boundaries=[v - 0.5 for v in unique_values] + [unique_values[-1] + 0.5], ncolors=len(unique_values))
        img = ax.imshow(
            root['data'][-1],
            cmap=cmap,
            norm=norm,
            interpolation='none',
            extent=[bounds[0], bounds[2], bounds[1], bounds[3]]  # west, east, south, north
        )
        
        # Add grid lines
        ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Format coordinate ticks more precisely
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
        
        # Configure plot
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Full Combined Dataset Land Cover 2023 (EPSG:4326)')
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP[val], 
                      markersize=10, label=f"{val}: {LABELS[val]}")
            for val in unique_values if val in LABELS
        ]
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=8
        )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'full_dataset_2023.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # Create figure for changes with same style as land cover map
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create a more informative colormap for changes
        from matplotlib.colors import LinearSegmentedColormap
        change_cmap = LinearSegmentedColormap.from_list(
            'change_gradient', 
            ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#d62828', '#6a0572'], 
            N=38  # One color for each possible change (max 38 years)
        )
        
        # Show changes with enhanced visualization
        img = ax.imshow(
            root['changes'],
            cmap=change_cmap,
            vmin=0,
            vmax=np.max(root['changes']),
            extent=[bounds[0], bounds[2], bounds[1], bounds[3]]  # west, east, south, north
        )
        
        # Add grid lines matching the land cover map
        ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Format coordinate ticks more precisely
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
        
        # Configure plot with more detailed information
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Full Dataset Land Cover Change Frequency 1985-2023 (EPSG:4326)', fontsize=14)
        
        # Add enhanced colorbar with more context
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('Number of Land Cover Changes (1985-2023)', fontsize=12)
        
        # Add annotation explaining the visualization
        max_changes = np.max(root['changes'])
        mean_changes = np.mean(root['changes'])
        ax.annotate(
            f"Analysis: Max changes = {max_changes}, Avg changes = {mean_changes:.1f}",
            xy=(0.02, 0.02),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            fontsize=10
        )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'full_dataset_changes.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        logging.info("Full dataset visualization complete")
        
    except Exception as e:
        logging.error(f"Error visualizing full dataset results: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    INPUT_BASE_DIR = '/srv/extrassd/mapbiomas_proc_zarr'  # Directory containing all the individual grid results
    OUTPUT_DIR = '/media/bndt-ai/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/combined_results'  # Directory for combined results
    
    try:
        logging.info("Starting combined analysis")
        
        # Combine all individual zarr datasets
        combined_zarr_path = combine_zarr_datasets(INPUT_BASE_DIR, OUTPUT_DIR)
        
        # Create visualizations for the full dataset
        visualize_full_results(combined_zarr_path, OUTPUT_DIR)
        
        # Create Sankey diagrams for the full dataset
        create_full_sankey_diagrams(combined_zarr_path, OUTPUT_DIR)
        
        logging.info(f"Combined analysis complete. Results saved to {OUTPUT_DIR}")
        
    except Exception as e:
        logging.critical(f"Combined analysis failed: {str(e)}", exc_info=True)