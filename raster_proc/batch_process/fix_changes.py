#!/usr/bin/env python3
"""
Simple changes recalculation script that directly reads existing Zarr data
and recalculates changes from the time series data.

This bypasses the tile processing issue.
"""

import zarr
import numpy as np
import sys
import os
import rasterio

def recalculate_changes_from_zarr(zarr_path, vrt_path):
    """Recalculate changes array directly from time series data"""
    print(f"\n=== Recalculating changes for: {zarr_path} ===")
    
    if not os.path.exists(zarr_path):
        print(f"ERROR: Zarr file does not exist: {zarr_path}")
        return False
        
    if not os.path.exists(vrt_path):
        print(f"ERROR: VRT file does not exist: {vrt_path}")
        return False
    
    try:
        # Open Zarr file
        root = zarr.open(zarr_path, mode='r+')  # Read-write mode
        
        # Get the window bounds and transform from metadata
        bounds = root.attrs['bounds']
        if isinstance(bounds, str):
            import json
            bounds = json.loads(bounds)
        
        window_transform = root.attrs['window_transform']
        if isinstance(window_transform, str):
            import json
            window_transform = json.loads(window_transform)
        
        # Create window for reading VRT
        height, width = root['first_year'].shape
        
        print(f"Zarr shape: {height} x {width}")
        print(f"Bounds: {bounds}")
        
        # Open VRT and create window
        with rasterio.open(vrt_path) as src:
            print(f"VRT bands: {src.count}")
            
            # Create window from bounds
            vrt_window = rasterio.windows.from_bounds(*bounds, src.transform)
            
            # Round window to integers and ensure it's within bounds
            vrt_window = rasterio.windows.Window(
                col_off=int(vrt_window.col_off),
                row_off=int(vrt_window.row_off),
                width=int(vrt_window.width),
                height=int(vrt_window.height)
            )
            
            print(f"VRT window: {vrt_window}")
            
            # Ensure window is within VRT bounds
            vrt_window = vrt_window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            
            if vrt_window.width == 0 or vrt_window.height == 0:
                print("ERROR: Window has no area after intersection")
                return False
            
            print(f"Final VRT window: {vrt_window}")
            
            # Read time series data and calculate changes
            changes = np.zeros((height, width), dtype='uint8')
            
            # Read first year
            previous_year = src.read(1, window=vrt_window)
            
            # Resize if needed to match Zarr dimensions
            if previous_year.shape != (height, width):
                print(f"Shape mismatch: VRT {previous_year.shape} vs Zarr {(height, width)}")
                # For small differences, just pad or crop
                if abs(previous_year.shape[0] - height) <= 1 and abs(previous_year.shape[1] - width) <= 1:
                    # Simple padding/cropping for 1-pixel differences
                    if previous_year.shape[0] < height:
                        previous_year = np.pad(previous_year, ((0, height - previous_year.shape[0]), (0, 0)), mode='edge')
                    elif previous_year.shape[0] > height:
                        previous_year = previous_year[:height, :]
                        
                    if previous_year.shape[1] < width:
                        previous_year = np.pad(previous_year, ((0, 0), (0, width - previous_year.shape[1])), mode='edge')
                    elif previous_year.shape[1] > width:
                        previous_year = previous_year[:, :width]
                else:
                    print(f"ERROR: Shape difference too large to handle: {previous_year.shape} vs {(height, width)}")
                    return False
            
            print(f"Processing {src.count} years...")
            
            # Process each subsequent year
            for band in range(2, src.count + 1):
                current_year = src.read(band, window=vrt_window)
                
                # Resize if needed
                if current_year.shape != (height, width):
                    # Simple padding/cropping for small differences
                    if abs(current_year.shape[0] - height) <= 1 and abs(current_year.shape[1] - width) <= 1:
                        if current_year.shape[0] < height:
                            current_year = np.pad(current_year, ((0, height - current_year.shape[0]), (0, 0)), mode='edge')
                        elif current_year.shape[0] > height:
                            current_year = current_year[:height, :]
                            
                        if current_year.shape[1] < width:
                            current_year = np.pad(current_year, ((0, 0), (0, width - current_year.shape[1])), mode='edge')
                        elif current_year.shape[1] > width:
                            current_year = current_year[:, :width]
                    else:
                        print(f"ERROR: Shape difference too large: {current_year.shape} vs {(height, width)}")
                        return False
                
                # Count changes
                changed_mask = (previous_year != current_year)
                changes[changed_mask] += 1
                
                # Count changes for this year
                year_changes = np.sum(changed_mask)
                total_pixels = changed_mask.size
                print(f"Year {band-1} -> {band}: {year_changes:,} pixels changed ({100*year_changes/total_pixels:.2f}%)")
                
                previous_year = current_year.copy()
            
            # Apply the same mask that was used originally (areas outside polygon)
            first_year_zarr = root['first_year'][:]
            mask = (first_year_zarr != 0)  # Assume 0 is fill value
            changes[~mask] = 0  # Set areas outside analysis to 0
            
            print(f"\nFinal changes summary:")
            print(f"Total pixels with changes: {np.sum(changes > 0):,}")
            print(f"Max changes per pixel: {np.max(changes)}")
            print(f"Changes distribution: {np.bincount(changes.flatten())}")
            
            # Update the changes array in Zarr
            if 'changes' in root:
                print(f"Updating existing changes array...")
                root['changes'][:] = changes
            else:
                print(f"Creating new changes array...")
                root.zeros('changes', shape=changes.shape, chunks=(256, 256), dtype='uint8')[:] = changes
            
            print(f"✅ Changes array updated successfully!")
            return True
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    vrt_path = "/ssdpro/mapbiomas10/mapbiomas10_1985_2024.vrt"
    
    # Process some test files
    test_files = [
        "/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/batch_process/indigenous_8/arara_1miLat_3.62685_miLon_53.32525_data.zarr",
        "/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/batch_process/indigenous_8/tupiniquim_1miLat_19.87174_miLon_40.17879_data.zarr"
    ]
    
    for zarr_file in test_files:
        if os.path.exists(zarr_file):
            success = recalculate_changes_from_zarr(zarr_file, vrt_path)
            if success:
                print(f"✅ Successfully updated {zarr_file}")
            else:
                print(f"❌ Failed to update {zarr_file}")
        else:
            print(f"⚠️  File not found: {zarr_file}")
