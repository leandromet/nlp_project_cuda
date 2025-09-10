#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main execution script for raster processing and visualization.

This script orchestrates the entire pipeline for processing land cover data,
extracting information by polygon, and creating visualizations.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import psutil
from config import setup_logging
from data_extraction import extract_grid_data_with_polygon
from visualization import create_visualizations
from sankey import create_sankey_diagram
from transition_viz import create_class_transition_visualization


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Process raster data and create visualizations for land cover analysis'
    )
    parser.add_argument(
        '--geojson', 
        required=True,
        help='Path to GeoJSON file containing polygons to process'
    )
    parser.add_argument(
        '--vrt', 
        required=True,
        help='Path to VRT file containing raster data'
    )
    parser.add_argument(
        '--output-dir', 
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--skip-extraction', 
        action='store_true',
        help='Skip data extraction and use existing Zarr files'
    )
    parser.add_argument(
        '--skip-visualization', 
        action='store_true',
        help='Skip visualization creation'
    )
    parser.add_argument(
        '--skip-sankey', 
        action='store_true',
        help='Skip Sankey diagram creation'
    )
    parser.add_argument(
        '--skip-transition-viz', 
        action='store_true',
        help='Skip transition visualization creation'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Validate inputs
    if not _validate_inputs(args):
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Log initial memory usage
        logging.info(f"Initial memory usage: {psutil.virtual_memory().used / 1024 / 1024:.1f} MB")
        
        # Step 1: Data extraction (if not skipped)
        if not args.skip_extraction:
            logging.info("Starting data extraction phase...")
            # Read GeoJSON to check number of features
            import geopandas as gpd
            gdf = gpd.read_file(args.geojson)
            logging.info(f"Found {len(gdf)} features in GeoJSON file")
            
            zarr_paths = []
            
            if len(gdf) == 1:
                # Single feature - process normally
                zarr_path, grid_output_dir = extract_grid_data_with_polygon(
                    vrt_path=args.vrt,
                    geojson_path=args.geojson,
                    output_base_dir=args.output_dir
                )
                zarr_paths = [zarr_path]
                logging.info(f"Data extraction completed. Created Zarr file: {zarr_path}")
            else:
                # Multiple features - process each individually
                logging.info(f"Processing {len(gdf)} features individually...")
                for idx, _ in gdf.iterrows():
                    try:
                        # Create temporary GeoJSON with single feature
                        single_gdf = gdf.iloc[[idx]]
                        temp_geojson = os.path.join(args.output_dir, f"temp_feature_{idx}.geojson")
                        single_gdf.to_file(temp_geojson, driver='GeoJSON')
                        
                        # Extract data for this feature
                        logging.info(f"Processing feature {idx + 1}/{len(gdf)}")
                        zarr_path, grid_output_dir = extract_grid_data_with_polygon(
                            vrt_path=args.vrt,
                            geojson_path=temp_geojson,
                            output_base_dir=args.output_dir
                        )
                        zarr_paths.append(zarr_path)
                        
                        # Clean up temporary file
                        os.remove(temp_geojson)
                        
                        logging.info(f"Feature {idx + 1} completed. Created: {os.path.basename(zarr_path)}")
                        
                    except Exception as e:
                        logging.error(f"Failed to process feature {idx + 1}: {e}")
                        # Clean up temporary file if it exists
                        temp_geojson = os.path.join(args.output_dir, f"temp_feature_{idx}.geojson")
                        if os.path.exists(temp_geojson):
                            os.remove(temp_geojson)
                        continue
                
                logging.info(f"Data extraction completed. Created {len(zarr_paths)} Zarr files.")
        else:
            # Find existing Zarr files
            zarr_paths = list(Path(args.output_dir).glob("*.zarr"))
            if not zarr_paths:
                logging.error("No Zarr files found in output directory and extraction was skipped")
                return 1
            logging.info(f"Found {len(zarr_paths)} existing Zarr files.")
        
        # Step 2: Create visualizations (if not skipped)
        if not args.skip_visualization:
            logging.info("Starting visualization phase...")
            for zarr_path in zarr_paths:
                try:
                    create_visualizations(str(zarr_path), args.output_dir, args.geojson)
                    logging.info(f"Created visualizations for {os.path.basename(str(zarr_path))}")
                except Exception as e:
                    logging.error(f"Failed to create visualizations for {os.path.basename(str(zarr_path))}: {e}")
        
        # Step 3: Create Sankey diagrams (if not skipped)
        if not args.skip_sankey:
            logging.info("Starting Sankey diagram creation...")
            for zarr_path in zarr_paths:
                try:
                    create_sankey_diagram(str(zarr_path), args.output_dir)
                    logging.info(f"Created Sankey diagram for {os.path.basename(str(zarr_path))}")
                except Exception as e:
                    logging.error(f"Failed to create Sankey diagram for {os.path.basename(str(zarr_path))}: {e}")
                    logging.exception("Sankey diagram error traceback:")
        
        # Step 4: Create transition visualizations (if not skipped)
        if not args.skip_transition_viz:
            logging.info("Starting transition visualization creation...")
            for zarr_path in zarr_paths:
                try:
                    # Create both treemap and stacked bar visualizations
                    create_class_transition_visualization(str(zarr_path), args.output_dir, vis_type='treemap')
                    create_class_transition_visualization(str(zarr_path), args.output_dir, vis_type='stacked_bar')
                    logging.info(f"Created transition visualizations for {os.path.basename(str(zarr_path))}")
                except Exception as e:
                    logging.error(f"Failed to create transition visualizations for {os.path.basename(str(zarr_path))}: {e}")
                    logging.exception("Transition visualization error traceback:")
        # Log final memory usage
        # Log final memory usage
        logging.info(f"Final memory usage: {psutil.virtual_memory().used / 1024 / 1024:.1f} MB")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        logging.exception("Full traceback:")
        return 1


def _validate_inputs(args):
    """Validate input arguments."""
    errors = []
    
    # Check if files exist
    if not os.path.exists(args.geojson):
        errors.append(f"GeoJSON file not found: {args.geojson}")
    
    if not os.path.exists(args.vrt):
        errors.append(f"VRT file not found: {args.vrt}")
    
    # Check if output directory is writable
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        test_file = os.path.join(args.output_dir, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        errors.append(f"Output directory not writable: {args.output_dir} ({e})")
    
    if errors:
        for error in errors:
            logging.error(error)
        return False
    
    return True


def run_batch_processing(geojson_path, vrt_path, output_dir, **kwargs):
    """
    Convenience function for running the entire pipeline programmatically.
    
    Args:
        geojson_path: Path to GeoJSON file
        vrt_path: Path to VRT file
        output_dir: Output directory
        **kwargs: Additional parameters (skip_extraction, skip_visualization, skip_sankey, skip_transition_viz, verbose)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Set default parameters
    params = {
        'skip_extraction': False,
        'skip_visualization': False,
        'skip_sankey': False,
        'skip_transition_viz': False,
        'verbose': False
    }
    params.update(kwargs)
    
    # Setup logging
    log_level = logging.DEBUG if params['verbose'] else logging.INFO
    setup_logging(log_level)
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Starting batch processing:")
        logging.info(f"  GeoJSON: {geojson_path}")
        logging.info(f"  VRT: {vrt_path}")
        logging.info(f"  Output: {output_dir}")
        
        # Data extraction
        if not params['skip_extraction']:
            # Read GeoJSON to check number of features
            import geopandas as gpd
            gdf = gpd.read_file(geojson_path)
            logging.info(f"Found {len(gdf)} features in GeoJSON file")
            
            zarr_paths = []
            
            if len(gdf) == 1:
                # Single feature - process normally
                zarr_path, grid_output_dir = extract_grid_data_with_polygon(
                    vrt_path=vrt_path,
                    geojson_path=geojson_path,
                    output_base_dir=output_dir
                )
                zarr_paths = [zarr_path]
            else:
                # Multiple features - process each individually
                logging.info(f"Processing {len(gdf)} features individually...")
                for idx, feature in gdf.iterrows():
                    try:
                        # Create temporary GeoJSON with single feature
                        single_gdf = gdf.iloc[[idx]]
                        temp_geojson = os.path.join(output_dir, f"temp_feature_{idx}.geojson")
                        single_gdf.to_file(temp_geojson, driver='GeoJSON')
                        
                        # Extract data for this feature
                        logging.info(f"Processing feature {idx + 1}/{len(gdf)}")
                        zarr_path, grid_output_dir = extract_grid_data_with_polygon(
                            vrt_path=vrt_path,
                            geojson_path=temp_geojson,
                            output_base_dir=output_dir
                        )
                        zarr_paths.append(zarr_path)
                        
                        # Clean up temporary file
                        os.remove(temp_geojson)
                        
                    except Exception as e:
                        logging.error(f"Failed to process feature {idx + 1}: {e}")
                        # Clean up temporary file if it exists
                        temp_geojson = os.path.join(output_dir, f"temp_feature_{idx}.geojson")
                        if os.path.exists(temp_geojson):
                            os.remove(temp_geojson)
                        continue
        else:
            zarr_paths = list(Path(output_dir).glob("*.zarr"))
        
        # Visualizations
        if not params['skip_visualization']:
            for zarr_path in zarr_paths:
                create_visualizations(str(zarr_path), output_dir, geojson_path)
        
        # Sankey diagrams
        if not params['skip_sankey']:
            for zarr_path in zarr_paths:
                create_sankey_diagram(str(zarr_path), output_dir)
        
        # Transition visualizations
        if not params['skip_transition_viz']:
            for zarr_path in zarr_paths:
                create_class_transition_visualization(str(zarr_path), output_dir, vis_type='treemap')
                create_class_transition_visualization(str(zarr_path), output_dir, vis_type='stacked_bar')
        
        logging.info("Batch processing completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        logging.exception("Full traceback:")
        return False


if __name__ == "__main__":
    sys.exit(main())
