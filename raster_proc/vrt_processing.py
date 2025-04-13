import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import zarr
import logging
import os

# Configuration
VRT_FILE = '/srv/extrassd/2025_mapbiomas/mapbiomas_coverage_1985_2023.vrt'
CHUNK_SIZE = 4096  # Optimal for ~1GB chunks (adjust based on your RAM)
OUTPUT_DIR = 'landcover_changes_zarr'  # Directory for Zarr storage
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_changes():
    try:
        logging.info("Opening VRT file: %s", VRT_FILE)
        with rasterio.open(VRT_FILE) as src:
            # Get raster dimensions
            width, height = src.width, src.height
            num_bands = src.count
            years = list(range(1985, 2024))  # 1985-2023
            logging.info("Raster dimensions: width=%d, height=%d, bands=%d", width, height, num_bands)
            
            # Create Zarr group (modern API)
            logging.info("Creating Zarr store at: %s", OUTPUT_DIR)
            root = zarr.open(OUTPUT_DIR, mode='w')
            
            # Initialize output arrays
            logging.info("Initializing output arrays")
            changes = root.zeros('changes', 
                               shape=(height, width), 
                               chunks=(CHUNK_SIZE, CHUNK_SIZE),
                               dtype='uint8')
            
            # Initialize transition matrix
            transition_matrix = root.zeros('transitions',
                                         shape=(256, 256),  # Max byte value
                                         dtype='uint64')
            
            # Process a few tiles only for testing
            logging.info("Starting limited chunk processing for testing")
            test_tiles = [(0, 0), (CHUNK_SIZE, 0), (0, CHUNK_SIZE)]  # Example tiles
            
            for y, x in tqdm(test_tiles, desc="Processing test tiles"):
                try:
                    # Read chunk for all years
                    window = Window(x, y, 
                                   min(CHUNK_SIZE, width - x), 
                                   min(CHUNK_SIZE, height - y))
                    
                    logging.debug("Processing window: x=%d, y=%d, width=%d, height=%d", 
                                 x, y, window.width, window.height)
                    data = src.read(window=window)
                    
                    # Compare consecutive years
                    for i in range(num_bands - 1):
                        band1, band2 = data[i], data[i+1]
                        
                        # Find changes
                        changed = band1 != band2
                        changes[y:y+window.height, x:x+window.width] += changed
                        
                        # Update transition matrix (sparse for memory)
                        unique_pairs, counts = np.unique(
                            np.vstack((band1[changed], band2[changed])).T,
                            axis=0,
                            return_counts=True
                        )
                        for (from_val, to_val), count in zip(unique_pairs, counts):
                            transition_matrix[from_val, to_val] += count
                            
                except Exception as e:
                    logging.error("Error processing window at x=%d, y=%d: %s", x, y, str(e))
                    raise
            
            # Save metadata
            logging.info("Saving metadata")
            root.attrs['years'] = years
            root.attrs['description'] = 'MapBiomas Brazil Land Cover Changes 1985-2023'
            
            logging.info("Processing complete")
            return root
            
    except Exception as e:
        logging.error("An error occurred during processing: %s", str(e))
        raise

if __name__ == '__main__':
    try:
        result = process_changes()
        logging.info("Processing complete. Results saved to: %s", OUTPUT_DIR)
        
        # Print some diagnostics
        total_changes = np.sum(result['changes'][:])
        logging.info("Total changed pixels in test area: %d", total_changes)
        
        # Show top 5 transitions
        transitions = result['transitions'][:]
        top_indices = np.unravel_index(np.argsort(-transitions, axis=None)[:5], transitions.shape)
        for i, (from_val, to_val) in enumerate(zip(*top_indices)):
            logging.info("Top transition %d: %d → %d (count: %d)", 
                        i+1, from_val, to_val, transitions[from_val, to_val])
            
    except Exception as e:
        logging.critical("Processing failed: %s", str(e))