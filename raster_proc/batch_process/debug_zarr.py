#!/usr/bin/env python3
"""
Debug script to check what's in the Zarr files
"""

import zarr
import numpy as np
import sys
import os

def debug_zarr_file(zarr_path):
    """Debug a Zarr file to see what data it contains"""
    print(f"\n=== Debugging Zarr file: {zarr_path} ===")
    
    if not os.path.exists(zarr_path):
        print(f"ERROR: File does not exist: {zarr_path}")
        return
    
    try:
        root = zarr.open(zarr_path, mode='r')
        print(f"Zarr file opened successfully")
        
        # List all arrays in the zarr file
        print(f"\nArrays in Zarr file:")
        for key in root.array_keys():
            arr = root[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            
        # Check changes specifically
        if 'changes' in root:
            changes = root['changes'][:]
            print(f"\n=== CHANGES ARRAY ANALYSIS ===")
            print(f"Shape: {changes.shape}")
            print(f"Dtype: {changes.dtype}")
            print(f"Min value: {np.min(changes)}")
            print(f"Max value: {np.max(changes)}")
            print(f"Unique values: {np.unique(changes)}")
            print(f"Value distribution: {np.bincount(changes.flatten())}")
            
            # Count non-zero values
            non_zero_count = np.sum(changes != 0)
            total_pixels = changes.size
            print(f"Non-zero pixels: {non_zero_count:,} out of {total_pixels:,} ({100*non_zero_count/total_pixels:.2f}%)")
            
            if non_zero_count > 0:
                print(f"Sample non-zero values: {changes[changes != 0][:20]}")
            else:
                print("WARNING: ALL VALUES ARE ZERO!")
                
            # Check if there are 255 values (outside polygon mask)
            mask_pixels = np.sum(changes == 255)
            if mask_pixels > 0:
                print(f"Pixels with value 255 (outside polygon): {mask_pixels:,}")
                
        else:
            print("ERROR: No 'changes' array found in Zarr file")
            
        # Check other arrays for comparison
        if 'first_year' in root and 'last_year' in root:
            first_year = root['first_year'][:]
            last_year = root['last_year'][:]
            
            print(f"\n=== FIRST/LAST YEAR COMPARISON ===")
            different_pixels = np.sum(first_year != last_year)
            valid_pixels = np.sum((first_year != 0) & (last_year != 0))
            print(f"Pixels different between first and last year: {different_pixels:,}")
            print(f"Valid pixels (non-zero in both years): {valid_pixels:,}")
            
            if valid_pixels > 0:
                print(f"Percentage different: {100*different_pixels/valid_pixels:.2f}%")
                
    except Exception as e:
        print(f"ERROR reading Zarr file: {str(e)}")

if __name__ == "__main__":
    # Test with a few example files
    test_files = [
        "/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/batch_process/indigenous_8/arara_1miLat_3.62685_miLon_53.32525_data.zarr",
        "/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/batch_process/indigenous_8/tupiniquim_1miLat_19.87174_miLon_40.17879_data.zarr"
    ]
    
    for test_file in test_files:
        debug_zarr_file(test_file)
