#!/usr/bin/env python3
"""
Test script to verify VRT reading and changes calculation
"""

import rasterio
import numpy as np
import sys
import os

def test_vrt_reading(vrt_path, sample_window=None):
    """Test reading from VRT file and calculating changes"""
    print(f"\n=== Testing VRT file: {vrt_path} ===")
    
    if not os.path.exists(vrt_path):
        print(f"ERROR: VRT file does not exist: {vrt_path}")
        return
    
    try:
        with rasterio.open(vrt_path) as src:
            print(f"VRT opened successfully")
            print(f"Bands: {src.count}")
            print(f"Size: {src.width} x {src.height}")
            print(f"CRS: {src.crs}")
            print(f"Data type: {src.dtypes[0]}")
            
            # Use a small sample window for testing
            if sample_window is None:
                # Use a small window in the center
                sample_window = rasterio.windows.Window(
                    col_off=src.width // 2,
                    row_off=src.height // 2,
                    width=100,
                    height=100
                )
            
            print(f"\nTesting with sample window: {sample_window}")
            
            # Read first few years
            years_to_test = min(5, src.count)  # Test first 5 years
            data = {}
            
            for band in range(1, years_to_test + 1):
                year_data = src.read(band, window=sample_window)
                data[band] = year_data
                unique_values = np.unique(year_data)
                print(f"Band {band}: shape={year_data.shape}, unique_values={unique_values[:10]}...")
                
            # Calculate changes between consecutive years
            print(f"\n=== CHANGES CALCULATION TEST ===")
            
            if len(data) >= 2:
                changes = np.zeros_like(data[1], dtype='uint8')
                
                previous_year = data[1].copy()
                for band in range(2, len(data) + 1):
                    current_year = data[band]
                    
                    # Track changes
                    changed_mask = (previous_year != current_year)
                    changes[changed_mask] += 1
                    
                    changed_pixels = np.sum(changed_mask)
                    total_pixels = changed_mask.size
                    print(f"Year {band-1} -> {band}: {changed_pixels:,} pixels changed ({100*changed_pixels/total_pixels:.2f}%)")
                    
                    previous_year = current_year.copy()
                
                print(f"\nFinal changes summary:")
                print(f"Total pixels: {changes.size:,}")
                print(f"Pixels with changes: {np.sum(changes > 0):,}")
                print(f"Max changes per pixel: {np.max(changes)}")
                print(f"Changes distribution: {np.bincount(changes.flatten())}")
                
                if np.sum(changes > 0) == 0:
                    print("WARNING: No changes detected!")
                    print("This could mean:")
                    print("  1. All years have identical data")
                    print("  2. The sample area has no land cover changes")
                    print("  3. There's an issue with the VRT structure")
                    
                    # Check if all years are identical
                    print("\nChecking if all years are identical...")
                    first_year = data[1]
                    all_identical = True
                    for band in range(2, len(data) + 1):
                        if not np.array_equal(first_year, data[band]):
                            all_identical = False
                            print(f"Year 1 vs {band}: NOT identical")
                            break
                    
                    if all_identical:
                        print("ERROR: All years contain identical data!")
                    else:
                        print("Years are different, but no consecutive changes detected")
                
            else:
                print("ERROR: Not enough bands to calculate changes")
                
    except Exception as e:
        print(f"ERROR reading VRT: {str(e)}")
        import traceback
        traceback.print_exc()

def test_polygon_area(vrt_path, geojson_path):
    """Test reading VRT data for a specific polygon area"""
    print(f"\n=== Testing VRT with polygon area ===")
    
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        
        # Read the polygon
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            print("ERROR: No polygons found in GeoJSON")
            return
            
        # Use first polygon
        polygon = gdf.iloc[0]
        bounds = polygon.geometry.bounds
        print(f"Polygon bounds: {bounds}")
        
        with rasterio.open(vrt_path) as src:
            # Create a window that covers the polygon
            window = rasterio.windows.from_bounds(*bounds, src.transform)
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            
            print(f"Window: {window}")
            
            if window.width > 0 and window.height > 0:
                # Test with a smaller sample if the window is large
                if window.width * window.height > 50000:  # 50k pixels max for test
                    # Take a sample from center of window
                    sample_window = rasterio.windows.Window(
                        col_off=window.col_off + window.width // 4,
                        row_off=window.row_off + window.height // 4,
                        width=min(200, window.width // 2),
                        height=min(200, window.height // 2)
                    )
                    print(f"Using sample window: {sample_window}")
                    test_vrt_reading(vrt_path, sample_window)
                else:
                    print(f"Using full polygon window")
                    test_vrt_reading(vrt_path, window)
            else:
                print("ERROR: Window has no area")
                
    except Exception as e:
        print(f"ERROR testing polygon area: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    vrt_path = "/ssdpro/mapbiomas10/mapbiomas10_1985_2024.vrt"
    
    # Test 1: Basic VRT reading with small sample
    test_vrt_reading(vrt_path)
    
    # Test 2: Test with actual polygon area
    geojson_path = "/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/batch_process/four_cases_complete.geojson"
    if os.path.exists(geojson_path):
        test_polygon_area(vrt_path, geojson_path)
    else:
        print(f"GeoJSON not found: {geojson_path}")
