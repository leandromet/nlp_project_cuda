import rasterio
from rasterio.windows import Window
import numpy as np
from rasterio.plot import show
import zarr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import logging
from tqdm import tqdm
import os
from matplotlib.colors import ListedColormap
import warnings
import json
from rasterio.transform import Affine
import plotly.graph_objects as go
from plotly.offline import plot
from shapely.geometry import Polygon

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grid_analysis.log'),
        logging.StreamHandler()
    ]
)

# Color mapping and labels
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

# List of polygons to process (coordinates in lon, lat format)
POLYGONS = [
    ((-74, 6), (-69, 6), (-69, 1), (-74, 1), (-74, 6)),
    ((-69, 6), (-64, 6), (-64, 1), (-69, 1), (-69, 6)),
    ((-64, 6), (-59, 6), (-59, 1), (-64, 1), (-64, 6)),
    ((-59, 6), (-54, 6), (-54, 1), (-59, 1), (-59, 6)),
    ((-54, 6), (-49, 6), (-49, 1), (-54, 1), (-54, 6)),
    ((-74, 1), (-69, 1), (-69, -4), (-74, -4), (-74, 1)),
    ((-69, 1), (-64, 1), (-64, -4), (-69, -4), (-69, 1)),
    ((-64, 1), (-59, 1), (-59, -4), (-64, -4), (-64, 1)),
    ((-59, 1), (-54, 1), (-54, -4), (-59, -4), (-59, 1)),
    ((-54, 1), (-49, 1), (-49, -4), (-54, -4), (-54, 1)),
    ((-49, 1), (-44, 1), (-44, -4), (-49, -4), (-49, 1)),
    ((-44, 1), (-39, 1), (-39, -4), (-44, -4), (-44, 1)),
    ((-39, 1), (-34, 1), (-34, -4), (-39, -4), (-39, 1)),
    ((-74, -4), (-69, -4), (-69, -9), (-74, -9), (-74, -4)),
    ((-69, -4), (-64, -4), (-64, -9), (-69, -9), (-69, -4)),
    ((-64, -4), (-59, -4), (-59, -9), (-64, -9), (-64, -4)),
    ((-59, -4), (-54, -4), (-54, -9), (-59, -9), (-59, -4)),
    ((-54, -4), (-49, -4), (-49, -9), (-54, -9), (-54, -4)),
    ((-49, -4), (-44, -4), (-44, -9), (-49, -9), (-49, -4)),
    ((-44, -4), (-39, -4), (-39, -9), (-44, -9), (-44, -4)),
    ((-39, -4), (-34, -4), (-34, -9), (-39, -9), (-39, -4)),
    ((-74, -9), (-69, -9), (-69, -14), (-74, -14), (-74, -9)),
    ((-69, -9), (-64, -9), (-64, -14), (-69, -14), (-69, -9)),
    ((-64, -9), (-59, -9), (-59, -14), (-64, -14), (-64, -9)),
    ((-59, -9), (-54, -9), (-54, -14), (-59, -14), (-59, -9)),
    ((-54, -9), (-49, -9), (-49, -14), (-54, -14), (-54, -9)),
    ((-49, -9), (-44, -9), (-44, -14), (-49, -14), (-49, -9)),
    ((-44, -9), (-39, -9), (-39, -14), (-44, -14), (-44, -9)),
    ((-39, -9), (-34, -9), (-34, -14), (-39, -14), (-39, -9)),
    ((-64, -14), (-59, -14), (-59, -19), (-64, -19), (-64, -14)),
    ((-59, -14), (-54, -14), (-54, -19), (-59, -19), (-59, -14)),
    ((-54, -14), (-49, -14), (-49, -19), (-54, -19), (-54, -14)),
    ((-49, -14), (-44, -14), (-44, -19), (-49, -19), (-49, -14)),
    ((-44, -14), (-39, -14), (-39, -19), (-44, -19), (-44, -14)),
    ((-59, -19), (-54, -19), (-54, -24), (-59, -24), (-59, -19)),
    ((-54, -19), (-49, -19), (-49, -24), (-54, -24), (-54, -19)),
    ((-49, -19), (-44, -19), (-44, -24), (-49, -24), (-49, -19)),
    ((-44, -19), (-39, -19), (-39, -24), (-44, -24), (-44, -19)),
    ((-59, -24), (-54, -24), (-54, -29), (-59, -29), (-59, -24)),
    ((-54, -24), (-49, -24), (-49, -29), (-54, -29), (-54, -24)),
    ((-49, -24), (-44, -24), (-44, -29), (-49, -29), (-49, -24)),
    ((-59, -29), (-54, -29), (-54, -34), (-59, -34), (-59, -29)),
    ((-54, -29), (-49, -29), (-49, -34), (-54, -34), (-54, -29))
]

def get_grid_name(polygon_coords):
    """Generate a grid name from polygon coordinates (e.g., grid_6n74w_1n69w)"""
    # Get min and max coordinates
    lons = [coord[0] for coord in polygon_coords]
    lats = [coord[1] for coord in polygon_coords]
    
    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)
    
    # Format latitude
    if max_lat >= 0:
        lat_str = f"{int(abs(max_lat))}n{int(abs(min_lat))}n"
    else:
        lat_str = f"{int(abs(min_lat))}s{int(abs(max_lat))}s"
    
    # Format longitude
    if min_lon >= 0:
        lon_str = f"{int(abs(min_lon))}e{int(abs(max_lon))}e"
    else:
        lon_str = f"{int(abs(max_lon))}w{int(abs(min_lon))}w"
    
    return f"grid_{lat_str}_{lon_str}"

def extract_grid_data(vrt_path, polygon_coords, output_dir):
    """Extract and analyze land cover changes for a specific polygon area."""
    grid_name = get_grid_name(polygon_coords)
    output_dir = os.path.join(output_dir, grid_name)
    
    # Get bounding box from polygon
    lons = [coord[0] for coord in polygon_coords]
    lats = [coord[1] for coord in polygon_coords]
    lon_range = (min(lons), max(lons))
    lat_range = (min(lats), max(lats))
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        with rasterio.open(vrt_path) as src:
            logging.info(f"Processing {grid_name}: Lon={lon_range}, Lat={lat_range}")
            
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
            
            # Create quick preview of area to confirm location
            preview_data = src.read(1, window=window)
            plt.figure(figsize=(8, 6))
            plt.imshow(preview_data)
            plt.title(f"Preview of {grid_name}")
            plt.colorbar(label='Land Cover Class')
            plt.savefig(os.path.join(output_dir, 'extraction_preview.png'))
            plt.close()
            
            # Read all years' data for this window
            data = src.read(window=window)
            window_transform = src.window_transform(window)
            
            # Create Zarr store
            zarr_path = os.path.join(output_dir, 'data.zarr')
            root = zarr.open(zarr_path, mode='w')
            
            # Store data with proper chunking
            data_array = root.zeros(
                'data',
                shape=data.shape,
                chunks=(1, 512, 512),
                dtype=data.dtype
            )
            data_array[:] = data
            
            # Calculate changes between years
            changes = np.zeros(data.shape[1:], dtype='uint8')
            transition_matrix = np.zeros((256, 256), dtype='uint64')
            
            for i in tqdm(range(data.shape[0]-1), desc="Calculating changes"):
                changed = data[i] != data[i+1]
                changes += changed
                
                unique_pairs, counts = np.unique(
                    np.vstack((data[i][changed], data[i+1][changed])).T,
                    axis=0,
                    return_counts=True
                )
                for (from_val, to_val), count in zip(unique_pairs, counts):
                    transition_matrix[from_val, to_val] += count
            
            # Store results
            root.zeros('changes', shape=changes.shape, chunks=(512, 512), dtype=changes.dtype)[:] = changes
            root.zeros('transitions', shape=transition_matrix.shape, dtype=transition_matrix.dtype)[:] = transition_matrix
            
            # Store critical metadata
            root.attrs.update({
                'window_transform': json.dumps(window_transform.to_gdal()),
                'crs': str(src.crs),
                'height': window.height,
                'width': window.width,
                'bounds': src.window_bounds(window),
                'lat_range': lat_range,
                'lon_range': lon_range,
                'grid_name': grid_name
            })
            
            logging.info(f"Processing complete for {grid_name}")
            return zarr_path
            
    except Exception as e:
        logging.error(f"Error processing {grid_name}: {str(e)}", exc_info=True)
        raise

def create_decadal_sankey_diagrams(root, output_dir):
    """Create Sankey diagrams showing both transitions and persistence of land cover classes."""
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    try:
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
                    title_text=f"{grid_name} Land Cover Changes {start_year}-{end_year}<br>"
                             f"(Showing persistence and transitions)",
                    font=dict(size=12, family="Arial"),
                    height=1200,
                    width=1600,
                    margin=dict(l=100, r=100, b=100, t=120, pad=20)
                )
                
                # Save files
                html_path = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.html')
                png_path = os.path.join(output_dir, f'transitions_{start_year}_{end_year}.png')
                
                plot(fig, filename=html_path, auto_open=False, include_plotlyjs='cdn')
                
                try:
                    fig.write_image(png_path, scale=2, engine="kaleido")
                    logging.info(f"Saved diagram to {png_path}")
                except Exception as e:
                    logging.warning(f"Could not save static image: {str(e)}")
                
                logging.info(f"Created diagram for {start_year}-{end_year} with persistence")
                
            except Exception as e:
                logging.error(f"Error creating {start_year}-{end_year} diagram: {str(e)}", exc_info=True)
                
    except Exception as e:
        logging.error(f"Error in decadal Sankey diagrams: {str(e)}", exc_info=True)
        raise

def visualize_results(zarr_path, output_dir):
    try:
        root = zarr.open(zarr_path, mode='r')
        grid_name = root.attrs.get('grid_name', 'unknown_grid')
        
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
        ax.set_title(f'{grid_name} Land Cover 2023 (EPSG:4326)')
        
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
            os.path.join(output_dir, f'{grid_name}_2023.png'),
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
        ax.set_title(f'{grid_name} Land Cover Change Frequency 1985-2023 (EPSG:4326)', fontsize=14)
        
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
            os.path.join(output_dir, f'{grid_name}_changes.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # Create decadal Sankey diagrams
        create_decadal_sankey_diagrams(root, output_dir)
        
        logging.info(f"Visualization complete for {grid_name}")
        
    except Exception as e:
        logging.error(f"Error visualizing results: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    VRT_FILE = '/srv/extrassd/2025_mapbiomas/mapbiomas_coverage_1985_2023.vrt'
    OUTPUT_BASE_DIR = 'grid_results'
    
    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        logging.info(f"Starting analysis for {len(POLYGONS)} polygons")
        
        for polygon_coords in POLYGONS:
            grid_name = get_grid_name(polygon_coords)
            output_dir = os.path.join(OUTPUT_BASE_DIR, grid_name)
            
            try:
                logging.info(f"Starting analysis for {grid_name}")
                zarr_path = extract_grid_data(VRT_FILE, polygon_coords, output_dir)
                visualize_results(zarr_path, output_dir)
                logging.info(f"Analysis complete for {grid_name}")
                
            except Exception as e:
                logging.error(f"Failed to process {grid_name}: {str(e)}", exc_info=True)
                continue
                
        logging.info(f"All analyses complete. Results saved to {OUTPUT_BASE_DIR}")
        
    except Exception as e:
        logging.critical(f"Analysis failed: {str(e)}", exc_info=True)