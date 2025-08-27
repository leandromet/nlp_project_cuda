#!/usr/bin/env python3
"""
Debug script to inspect actual VRT data for a specific area
"""

import rasterio
import numpy as np
import sys
import geopandas as gpd
from rasterio.windows import Window

def debug_vrt_data(vrt_path, geojson_path):
    """Debug VRT data for a specific polygon area"""
    print(f"=== Debugging VRT data ===")
    print(f"VRT: {vrt_path}")
    print(f"GeoJSON: {geojson_path}")
    
    # Read the polygon
    gdf = gpd.read_file(geojson_path)
    print(f"Found {len(gdf)} features in GeoJSON")
    
    # Use the first feature
    feature = gdf.iloc[0]
    bounds = feature.geometry.bounds
    print(f"Feature bounds: {bounds}")
    
    with rasterio.open(vrt_path) as src:
        print(f"VRT info:")
        print(f"  Shape: {src.shape}")
        print(f"  Bands: {src.count}")
        print(f"  CRS: {src.crs}")
        print(f"  Transform: {src.transform}")
        
        # Get window for the polygon bounds
        window = src.window(*bounds)
        print(f"Window for polygon: {window}")
        
        # Read a small sample (just first few years)
        sample_years = min(5, src.count)
        print(f"\nReading first {sample_years} years...")
        
        for year_idx in range(1, sample_years + 1):
            try:
                year_data = src.read(year_idx, window=window)
                unique_vals, counts = np.unique(year_data, return_counts=True)
                print(f"  Year {year_idx}: shape={year_data.shape}, unique values={len(unique_vals)}")
                print(f"    Value distribution: {dict(zip(unique_vals[:10], counts[:10]))}")  # Show first 10
                
                if year_idx == 1:
                    first_year = year_data
                elif year_idx == sample_years:
                    last_year = year_data
                    
            except Exception as e:
                print(f"  Year {year_idx}: ERROR - {str(e)}")
        
        # Compare first vs last year
        if 'first_year' in locals() and 'last_year' in locals():
            print(f"\nComparing first vs last year:")
            different_pixels = np.sum(first_year != last_year)
            valid_pixels = np.sum((first_year != 0) & (last_year != 0))
            total_pixels = first_year.size
            
            print(f"  Total pixels: {total_pixels:,}")
            print(f"  Valid pixels (non-zero): {valid_pixels:,}")
            print(f"  Different pixels: {different_pixels:,}")
            if valid_pixels > 0:
                print(f"  Change percentage: {100*different_pixels/valid_pixels:.2f}%")

if __name__ == "__main__":
    vrt_path = "/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/mapbiomas_coverage_1985_2023.vrt"
    geojson_path = "/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/batch_process/four_cases_complete.geojson"
    
    debug_vrt_data(vrt_path, geojson_path)
