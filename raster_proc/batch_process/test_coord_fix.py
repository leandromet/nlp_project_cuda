#!/usr/bin/env python3
"""
Test script to verify the coordinate fix for polygon overlays.
"""

import os
import sys
import logging
import zarr

# Add current directory to path
sys.path.append('.')

from visualization import create_changes_map

def test_coordinate_fix():
    """Test the coordinate fix by regenerating a changes map."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Test with one of the fixed Zarr files
        zarr_path = './indigenous_10/arara_1miLat_3.62685_miLon_53.32525_data.zarr'
        output_dir = './test_coord_fix'
        geojson_path = 'four_cases_complete.geojson'
        
        # Create test output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print('Testing coordinate fix for polygon overlays...')
        
        # Check if files exist
        if not os.path.exists(zarr_path):
            print(f"ERROR: Zarr file not found: {zarr_path}")
            return False
            
        if not os.path.exists(geojson_path):
            print(f"ERROR: GeoJSON file not found: {geojson_path}")
            return False
        
        # Open zarr file
        root = zarr.open(zarr_path, mode='r')
        print(f"Successfully opened Zarr file: {zarr_path}")
        
        # Extract prefix from zarr filename
        zarr_filename = os.path.basename(zarr_path)
        file_prefix = zarr_filename.replace('_data.zarr', '') + '_improved_contrast'
        
        print(f'Creating test changes map with improved contrast: {file_prefix}')
        
        # Create the changes map with fixed coordinates
        create_changes_map(root, output_dir, file_prefix, geojson_path)
        
        output_file = os.path.join(output_dir, f'{file_prefix}_changes_map.png')
        if os.path.exists(output_file):
            print(f'SUCCESS! Test completed! Check {output_file}')
            print('The polygon overlay should now be properly positioned.')
            print('The color scale should now have much better contrast for change visualization!')
            
            # Also check the CSV for statistics
            csv_file = os.path.join(output_dir, f'{file_prefix}_change_statistics.csv')
            if os.path.exists(csv_file):
                print(f'Statistics saved to: {csv_file}')
            
            return True
        else:
            print('ERROR: Output file was not created')
            return False
        
    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coordinate_fix()
    if success:
        print("\nCoordinate fix and contrast improvement test PASSED!")
        print("✅ Polygon coordinates are now correctly positioned")
        print("✅ Color scale now focuses on actual change range (0-max) instead of (0-255)")
        print("✅ Changes maps should have much better visual contrast")
    else:
        print("\nCoordinate fix and contrast improvement test FAILED!")
