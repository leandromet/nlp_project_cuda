# -*- coding: utf-8 -*-
"""
Utilities module for raster processing.

Contains helper functions for reading raster data, processing tiles,
and calculating changes and transitions.
"""

import numpy as np
import rasterio
import logging
import time
from rasterio.windows import Window
from config import MAX_READ_RETRIES, RETRY_DELAY


def robust_read(src, band, window, retries=MAX_READ_RETRIES):
    """Attempt to read a tile with retries on failure."""
    for attempt in range(retries):
        try:
            start_time = time.time()
            data = src.read(band, window=window, fill_value=0)
            read_time = time.time() - start_time
            
            # Log slow reads
            if read_time > 5.0:  # More than 5 seconds
                logging.warning(f"Slow read detected: {read_time:.1f}s for band {band}, "
                              f"window {window.width}x{window.height}")
            
            return data
            
        except Exception as e:
            if attempt < retries - 1:
                logging.warning(f"Read attempt {attempt + 1} failed: {str(e)}, retrying...")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logging.error(f"All read attempts failed for band {band}: {str(e)}")
                # Return zeros array if all attempts fail
                return np.zeros((window.height, window.width), dtype=np.uint8)
    
    # This should never be reached
    return np.zeros((window.height, window.width), dtype=np.uint8)


def process_tile(task):
    """Process a single tile for changes and transitions between all years."""
    y, x, path, window, temp_dir = task
    try:
        with rasterio.open(path) as src:
            # Read first year as reference
            first_year = robust_read(src, 1, window)
            changes = np.zeros_like(first_year, dtype='uint8')
            transition_matrix = np.zeros((256, 256), dtype='uint64')
            
            # Process subsequent years
            for year_idx in range(2, src.count + 1):
                current_year = robust_read(src, year_idx, window)
                
                # Track changes
                changed_mask = (first_year != current_year)
                changes[changed_mask] += 1
                
                # Update transition matrix
                for from_cls in np.unique(first_year):
                    from_mask = (first_year == from_cls)
                    for to_cls in np.unique(current_year[from_mask]):
                        count = np.sum((first_year == from_cls) & (current_year == to_cls))
                        transition_matrix[from_cls, to_cls] += count
                
                # Update reference for next iteration
                first_year = current_year
            
            return changes, transition_matrix, (y, x), False
            
    except Exception as e:
        logging.warning(f"Tile {y},{x} failed: {str(e)}")
        return None, None, (y, x), True


def calculate_class_persistence(src, window):
    """Calculate which pixels remained in the same class for all years."""
    # Read first year as reference
    reference = robust_read(src, 1, window)
    persistence = np.ones_like(reference, dtype=bool)
    
    # Compare against all subsequent years
    for year_idx in range(2, src.count + 1):
        current = robust_read(src, year_idx, window)
        persistence &= (reference == current)
    
    # Return both the persistence mask and the reference classes
    return persistence, reference


def calculate_changes_tile(args):
    """Calculate changes for a single tile with robust error handling."""
    tile_idx, vrt_path, window, temp_dir = args
    y, x = tile_idx
    
    changes = np.zeros((window.height, window.width), dtype='uint8')
    transition_matrix = np.zeros((256, 256), dtype='uint64')
    class_persistence = np.zeros((256,), dtype='uint64')  # Persistent pixels per class
    class_initial = np.zeros((256,), dtype='uint64')      # Initial pixels per class
    failed = False

    try:
        with rasterio.open(vrt_path) as src:
            # Read first year with retries
            try:
                first_year = robust_read(src, 1, window)
                
                # Calculate initial class counts
                unique_initial, counts_initial = np.unique(first_year, return_counts=True)
                for cls, count in zip(unique_initial, counts_initial):
                    if cls < 256:
                        class_initial[cls] = count
                        
            except Exception as e:
                logging.error(f"Failed to read first year for tile {y},{x}: {str(e)}")
                failed = True
                return changes, transition_matrix, class_persistence, class_initial, (y, x), failed
            
            # Calculate persistence for the entire time series
            persistence_mask, reference_classes = calculate_class_persistence(src, window)
            
            # Count persistent pixels by class
            for cls in np.unique(reference_classes):
                if cls < 256:
                    class_persistence[cls] = np.sum((reference_classes == cls) & persistence_mask)
            
            # Process subsequent years for transitions
            previous_year = first_year.copy()
            for year_idx in range(2, src.count + 1):
                try:
                    current_year = robust_read(src, year_idx, window)
                    
                    # Track changes
                    changed_mask = (previous_year != current_year)
                    changes[changed_mask] += 1
                    
                    # Update transition matrix
                    for from_cls in np.unique(previous_year):
                        if from_cls < 256:
                            from_mask = (previous_year == from_cls)
                            for to_cls in np.unique(current_year[from_mask]):
                                if to_cls < 256:
                                    count = np.sum((previous_year == from_cls) & (current_year == to_cls))
                                    transition_matrix[from_cls, to_cls] += count
                    
                    previous_year = current_year.copy()
                    
                except Exception as e:
                    logging.error(f"Failed to read year {year_idx} for tile {y},{x}: {str(e)}")
                    failed = True
                    break
            
            return changes, transition_matrix, class_persistence, class_initial, (y, x), failed

    except Exception as e:
        logging.error(f"Critical error processing tile {y},{x}: {str(e)}")
        return None, None, None, None, (y, x), True


def generate_local_grid_name(gdf):
    """Generate grid name from terrai_nom field or use simple fallback."""
    # Try to get terrai_nom field as prefix
    if not gdf.empty and 'terrai_nom' in gdf.columns:
        terrai_nom = str(gdf.iloc[0]['terrai_nom']).replace(' ', '_').replace('/', '_').replace('-', '_')
        return terrai_nom
    else:
        # Fallback to simple numbered prefix if terrai_nom doesn't exist
        return f"grid_{len(gdf)}"
