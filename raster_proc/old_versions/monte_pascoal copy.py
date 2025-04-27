import rasterio
from rasterio.windows import Window
import numpy as np
import zarr
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import os
from matplotlib.colors import ListedColormap
import contextily as cx
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monte_pascoal_analysis.log'),
        logging.StreamHandler()
    ]
)

def extract_monte_pascoal(vrt_path, output_dir):
    """Extract and analyze land cover changes for Monte Pascoal area."""
    # Monte Pascoal coordinates - verified against VRT bounds
    lat_range = (-17.21, -16.2)  # South to North
    lon_range = (-39.5, -38.5)   # West to East
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        with rasterio.open(vrt_path) as src:
            logging.info(f"VRT bounds: {src.bounds}")
            logging.info(f"VRT transform: {src.transform}")
            
            # Use correct method for coordinate conversion
            # This handles the axis order properly in WGS84
            ul_row, ul_col = src.index(lon_range[0], lat_range[1])  # Upper left (west, north)
            lr_row, lr_col = src.index(lon_range[1], lat_range[0])  # Lower right (east, south)
            
            # Create window with correct bounds
            window = Window.from_slices(
                rows=(ul_row, lr_row),  # Row indices 
                cols=(ul_col, lr_col)   # Column indices
            )
            
            logging.info(f"Extracting window: {window}")
            
            # Read all years' data for this window
            data = src.read(window=window)
            years = list(range(1985, 2024))
            
            # Create Zarr store using proper modern API - FIXED
            zarr_path = os.path.join(output_dir, 'data.zarr')
            logging.info(f"Extracting window: {window}")
            
            # Define the root Zarr group
            root = zarr.open(zarr_path, mode='w')
            
            # Store metadata after defining window, transform, and crs
            root.attrs.update({
                'years': years,
                'description': 'Monte Pascoal Land Cover 1985-2023',
                'window': str(window),
                'transform': str(src.transform),
                'crs': str(src.crs),
                'lat_range': lat_range,
                'lon_range': lon_range
            })
            
            
            
            # Store the multi-year data
            data_array = root.create(
                'data',
                shape=data.shape,
                chunks=(1, 512, 512),
                dtype=data.dtype
            )
            data_array[:] = data
            
            # Store metadata
            root.attrs.update({
                'years': years,
                'description': 'Monte Pascoal Land Cover 1985-2023',
                'window': str(window),
                'transform': str(src.transform),
                'crs': str(src.crs)
            })
            
            # Calculate changes between years
            changes = np.zeros(data.shape[1:], dtype='uint8')
            transition_matrix = np.zeros((256, 256), dtype='uint64')
            
            for i in tqdm(range(len(years)-1), desc="Calculating changes"):
                changed = data[i] != data[i+1]
                changes += changed
                
                # Update transition matrix
                unique_pairs, counts = np.unique(
                    np.vstack((data[i][changed], data[i+1][changed])).T,
                    axis=0,
                    return_counts=True
                )
                for (from_val, to_val), count in zip(unique_pairs, counts):
                    transition_matrix[from_val, to_val] += count
            
            # Store results
            root.create(
                'changes',
                shape=changes.shape,
                chunks=(512, 512),
                dtype=changes.dtype
            )[:] = changes
            
            root.create(
                'transitions',
                shape=transition_matrix.shape,
                dtype=transition_matrix.dtype
            )[:] = transition_matrix
            
            logging.info("Processing complete")
            
            # return root
            
    except Exception as e:
        logging.error(f"Error processing Monte Pascoal: {str(e)}", exc_info=True)
        raise

def visualize_results(output_dir):
    try:
        root = zarr.open(os.path.join(output_dir, 'data.zarr'), mode='r')

        # Get the geographic bounds from the metadata
        lat_range = root.attrs.get('lat_range', (-17.21, -16.2))
        lon_range = root.attrs.get('lon_range', (-39.5, -38.5))
        
        # Calculate extent for plotting (left, right, bottom, top)
        extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
        
        # Define a custom colormap for 2023 data
        color_map = {
            3: "#1f8d49", 4: "#7dc975", 5: "#04381d", 6: "#007785", 9: "#7a5900",
            11: "#519799", 12: "#d6bc74", 15: "#edde8e", 20: "#db7093", 21: "#ffefc3",
            23: "#ffa07a", 24: "#d4271e", 25: "#db4d4f", 29: "#ffaa5f", 30: "#9c0027",
            31: "#091077", 32: "#fc8114", 33: "#2532e4", 35: "#9065d0", 39: "#f5b3c8",
            40: "#c71585", 41: "#f54ca9", 46: "#d68fe2", 47: "#9932cc", 48: "#e6ccff",
            49: "#02d659", 50: "#ad5100", 62: "#ff69b4"
        }
        
        # Create a colormap and legend based on the fixed color codes and labels
        unique_values = sorted(color_map.keys())
        colors = [color_map[val] for val in unique_values]
        labels = [
            "Forest Formation", "Savanna Formation", "Mangrove", "Floodable Forest",
            "Forest Plantation", "Wetland", "Grassland", "Pasture", "Sugar Cane",
            "Mosaic of Uses", "Beach, Dune and Sand Spot", "Urban Area",
            "Other non Vegetated Areas", "Rocky Outcrop", "Mining", "Aquaculture",
            "Hypersaline Tidal Flat", "River, Lake and Ocean", "Palm Oil", "Soybean",
            "Rice", "Other Temporary Crops", "Coffee", "Citrus", "Other Perennial Crops",
            "Wooded Sandbank Vegetation", "Herbaceous Sandbank Vegetation", "Cotton"
        ]
        cmap = ListedColormap(colors)

        # Create a legend for the plot
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
            for color, label in zip(colors, labels)
        ]

        # Visualize the most recent year with coordinates
                # Replace the problematic section in the visualization function
        # Visualize the most recent year with coordinates
        plt.figure(figsize=(14, 12))
        ax = plt.gca()
        
        try:
            # Add basemap first
            cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.Esri.WorldImagery)
            
            # Then overlay land cover (transparent)
            land_cover = ax.imshow(root['data'][-1], cmap=cmap, extent=extent, alpha=0.7)
            
            # Add colorbar once
            cbar = plt.colorbar(land_cover, extend='both', ticks=unique_values)
            cbar.set_label('Land Cover Class', fontsize=12)
            cbar.ax.set_yticklabels([f"Class {val}" for val in unique_values])
            
        except ImportError:
            # Fallback if contextily isn't available
            img = plt.imshow(root['data'][-1], cmap=cmap, extent=extent)
            # Ensure 'img' is defined before using it
            img = plt.imshow(root['data'][-1], cmap=cmap, extent=extent, alpha=0.7)
            # Ensure 'img' is defined before using it
            img = plt.imshow(root['data'][-1], cmap=cmap, extent=extent, alpha=0.7)
            cbar = plt.colorbar(img, extend='both', ticks=unique_values)
            cbar.set_label('Land Cover Class', fontsize=12)
            cbar.ax.set_yticklabels([f"Class {val}" for val in unique_values])
            logging.warning("Contextily not installed, skipping basemap")
        
        # Add landmarks
        landmarks = {
            "Monte Pascoal Peak": (-16.8578, -39.0737),
            "Caraíva": (-16.8081, -39.1497),
            "Barra do Cahy": (-16.8806, -39.1539)
        }
        for name, (lat, lon) in landmarks.items():
            plt.plot(lon, lat, 'k*', markersize=10)
            plt.text(lon, lat, f" {name}", fontsize=10, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Add coordinate labels and grid
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.grid(alpha=0.3)
        
        # Format coordinate ticks
        lon_ticks = np.linspace(extent[0], extent[1], 5)
        lat_ticks = np.linspace(extent[2], extent[3], 5)
        plt.xticks(lon_ticks, [f"{x:.4f}°" for x in lon_ticks])
        plt.yticks(lat_ticks, [f"{y:.4f}°" for y in lat_ticks])
        
        plt.title('Monte Pascoal Land Cover 2023 (EPSG:4326)', fontsize=14)
        cbar = plt.colorbar(img, extend='both', ticks=unique_values)
        cbar.set_label('Land Cover Class', fontsize=12)
        cbar.ax.set_yticklabels([f"Class {val}" for val in unique_values])
        
        # Add legend in a separate box outside the main plot
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                   loc='lower left', fontsize=10, title="Land Cover Classes")
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'monte_pascoal_2023.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        # Visualize change frequency with coordinates
        plt.figure(figsize=(14, 12))
        img = plt.imshow(root['changes'], cmap='YlOrRd', vmin=0, extent=extent)
        
        # Add coordinate labels and grid
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.grid(alpha=0.3)
        
        # Format coordinate ticks
        plt.xticks(lon_ticks, [f"{x:.4f}°" for x in lon_ticks])
        plt.yticks(lat_ticks, [f"{y:.4f}°" for y in lat_ticks])
        
        plt.title('Change Frequency 1985-2023 (EPSG:4326)', fontsize=14)
        cbar = plt.colorbar(img)
        cbar.set_label('Number of Changes', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'monte_pascoal_changes.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        # Print top transitions
        transitions = root['transitions'][:]
        top_n = 5
        flat_indices = np.argsort(-transitions, axis=None)[:top_n]
        top_indices = np.unravel_index(flat_indices, transitions.shape)

        logging.info("\nTop 5 Land Cover Transitions:")
        for i, (from_class, to_class) in enumerate(zip(*top_indices)):
            logging.info(f"{i+1}. Class {from_class} → Class {to_class}: {transitions[from_class, to_class]} pixels")

    except Exception as e:
        logging.error(f"Error visualizing results: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    VRT_FILE = '/srv/extrassd/2025_mapbiomas/mapbiomas_coverage_1985_2023.vrt'
    OUTPUT_DIR = 'monte_pascoal_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        logging.info("Starting Monte Pascoal analysis")
        store = extract_monte_pascoal(VRT_FILE, OUTPUT_DIR)
        visualize_results(OUTPUT_DIR)
        logging.info(f"Analysis complete. Results saved to {OUTPUT_DIR}")
        
    except Exception as e:
        logging.critical(f"Analysis failed: {str(e)}", exc_info=True)