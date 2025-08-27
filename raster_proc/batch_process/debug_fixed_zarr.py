#!/usr/bin/env python3
"""
Debug script to verify the fixed Zarr data has proper changes distribution.
"""

import numpy as np
import zarr

def debug_zarr_file(zarr_path):
    """Debug a Zarr file to show changes array statistics."""
    print(f"=== Debugging Zarr file: {zarr_path} ===")
    
    try:
        # Open Zarr file
        zarr_store = zarr.open(zarr_path, mode='r')
        print("Zarr file opened successfully")
        
        # List arrays
        print("\nArrays in Zarr file:")
        for key in zarr_store.keys():
            array = zarr_store[key]
            print(f"  {key}: shape={array.shape}, dtype={array.dtype}")
        
        # Analyze changes array
        changes = zarr_store['changes'][:]
        print(f"\n=== CHANGES ARRAY ANALYSIS ===")
        print(f"Shape: {changes.shape}")
        print(f"Dtype: {changes.dtype}")
        print(f"Min value: {np.min(changes)}")
        print(f"Max value: {np.max(changes)}")
        print(f"Unique values: {np.unique(changes)[:20]}...")  # Show first 20 unique values
        
        # Value distribution
        unique_vals, counts = np.unique(changes, return_counts=True)
        print(f"Value distribution (first 10):")
        for val, count in list(zip(unique_vals, counts))[:10]:
            percentage = 100.0 * count / changes.size
            print(f"  Value {val}: {count:,} pixels ({percentage:.2f}%)")
        
        # Key statistics
        total_pixels = changes.size
        pixels_with_255 = np.sum(changes == 255)
        pixels_with_changes = np.sum((changes > 0) & (changes < 255))
        pixels_unchanged = np.sum(changes == 0)
        
        print(f"\n=== KEY STATISTICS ===")
        print(f"Total pixels: {total_pixels:,}")
        print(f"Pixels with 255 (fill value): {pixels_with_255:,} ({100*pixels_with_255/total_pixels:.2f}%)")
        print(f"Pixels with actual changes (1-254): {pixels_with_changes:,} ({100*pixels_with_changes/total_pixels:.2f}%)")
        print(f"Pixels unchanged (0): {pixels_unchanged:,} ({100*pixels_unchanged/total_pixels:.2f}%)")
        
        # Also check first/last year for comparison
        if 'first_year' in zarr_store and 'last_year' in zarr_store:
            first_year = zarr_store['first_year'][:]
            last_year = zarr_store['last_year'][:]
            
            # Compare first and last year to verify data makes sense
            different_pixels = np.sum(first_year != last_year)
            valid_pixels = np.sum((first_year > 0) & (last_year > 0))
            
            print(f"\n=== FIRST/LAST YEAR COMPARISON ===")
            print(f"Pixels different between first and last year: {different_pixels:,}")
            print(f"Valid pixels (non-zero in both years): {valid_pixels:,}")
            if valid_pixels > 0:
                print(f"Percentage different: {100*different_pixels/valid_pixels:.2f}%")
        
    except Exception as e:
        print(f"Error debugging Zarr file: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Debug a few representative files from the fixed data
    zarr_files = [
        "/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/batch_process/indigenous_10/arara_1miLat_3.62685_miLon_53.32525_data.zarr",
        "/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/batch_process/indigenous_10/tupiniquim_1miLat_19.87174_miLon_40.17879_data.zarr"
    ]
    
    for zarr_path in zarr_files:
        debug_zarr_file(zarr_path)
        print("\n" + "="*80 + "\n")
