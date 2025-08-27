# -*- coding: utf-8 -*-
"""
Data extraction module for raster processing.

Handles polygon-based data extraction from VRT files and Zarr storage.
"""

import os
import tempfile
import json
import time
import logging
import numpy as np
import zarr
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from rasterio.features import rasterize
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import (
    VRT_BLOCK_SIZE, PROCESSING_TILE_SIZE, MAX_WORKERS, FILL_VALUE
)
from utils import calculate_changes_tile, generate_local_grid_name


def extract_grid_data_with_polygon(vrt_path, geojson_path, output_base_dir):
    """Extract data for a specific polygon from a GeoJSON file, masking areas outside the polygon with zeros."""
    
    try:
        os.makedirs(output_base_dir, exist_ok=True)
        logging.info("Starting robust analysis with visualization")

        # Load the GeoJSON file
        gdf = gpd.read_file(geojson_path)
        gdf = gdf.to_crs(epsg=4326)  # Ensure WGS84 coordinates

        # Extract the polygon coordinates from the GeoJSON
        if not gdf.empty and 'geometry' in gdf.columns:
            geometry = gdf.iloc[0].geometry
            if geometry.type == 'MultiPolygon':
                polygons = list(geometry.geoms)
            else:
                polygons = [geometry]
        else:
            raise ValueError("GeoJSON file is empty or does not contain valid geometry.")

        # Setup output directory - use prefix for files, not nested folders
        grid_name = generate_local_grid_name(gdf)
        grid_output_dir = output_base_dir  # Use the main output directory directly
        os.makedirs(grid_output_dir, exist_ok=True)
        logging.info(f"Using output directory: {grid_output_dir}")
        logging.info(f"Files will be prefixed with: {grid_name}")
        
        # Also store terrai_nom for later use if available
        terrai_nom = None
        if not gdf.empty and 'terrai_nom' in gdf.columns:
            terrai_nom = str(gdf.iloc[0]['terrai_nom'])
            
        # Store the geometry as WKT for visualization overlay
        feature_geometry_wkt = None
        if not gdf.empty and 'geometry' in gdf.columns:
            feature_geometry = gdf.iloc[0].geometry
            feature_geometry_wkt = feature_geometry.wkt

        with rasterio.open(vrt_path) as src:
            logging.info(f"Dataset info - Width: {src.width}, Height: {src.height}, Bands: {src.count}")
            transform = src.transform

            # Get bounding box of all polygons
            bounds = gdf.total_bounds
            min_lon, min_lat, max_lon, max_lat = bounds

            # Convert geographic coordinates to pixel coordinates using floating-point precision
            ul_col = (min_lon - transform.c) / transform.a
            ul_row = (max_lat - transform.f) / transform.e  # max_lat for upper row
            lr_col = (max_lon - transform.c) / transform.a
            lr_row = (min_lat - transform.f) / transform.e  # min_lat for lower row

            # Ensure window is within bounds
            ul_row = max(0.0, ul_row)
            ul_col = max(0.0, ul_col)
            lr_row = min(src.height, lr_row)
            lr_col = min(src.width, lr_col)

            # Ensure the bounding box intersects with the raster bounds
            if ul_row >= lr_row or ul_col >= lr_col:
                raise ValueError("Aligned window size is invalid: width and height must be greater than 0")

            full_window = Window.from_slices(
                rows=(int(ul_row), int(lr_row)),
                cols=(int(ul_col), int(lr_col))
            )

            logging.info(f"Processing {grid_name} with aligned window size: {full_window.height}x{full_window.width}")
            
            # Check if the window is too large and warn the user
            total_pixels = full_window.height * full_window.width
            if total_pixels > 50_000_000:  # 50 million pixels
                logging.warning(f"Large processing area detected: {total_pixels:,} pixels. This may take a very long time.")
                logging.warning(f"Consider using a smaller polygon or increasing PROCESSING_TILE_SIZE in config.")
            
            # Estimate processing time
            estimated_tiles = ((full_window.height + PROCESSING_TILE_SIZE - 1) // PROCESSING_TILE_SIZE) * \
                             ((full_window.width + PROCESSING_TILE_SIZE - 1) // PROCESSING_TILE_SIZE)
            logging.info(f"Estimated tiles to process: {estimated_tiles}")
            
            if estimated_tiles > 500:
                logging.warning(f"Large number of tiles ({estimated_tiles}). This may take several hours.")
                logging.warning("Consider:")
                logging.warning("  1. Using a smaller polygon area")
                logging.warning("  2. Increasing PROCESSING_TILE_SIZE in config.py")
                logging.warning("  3. Reducing MAX_WORKERS if memory is limited")
                logging.warning("Processing will continue automatically...")

            # Create a mask for the polygon(s)
            mask_shape = (full_window.height, full_window.width)
            
            # Get the proper transform for the window
            transform_window = src.window_transform(full_window)
            
            # Rasterize polygons using proper geometric shapes
            polygon_shapes = [(geom, 1) for geom in polygons]
            
            # Rasterize with fill=0 and default_value=1
            mask = rasterize(
                polygon_shapes,
                out_shape=mask_shape,
                transform=transform_window,
                fill=0,
                default_value=1,
                dtype=np.uint8
            )

            # Convert mask to boolean after all polygons are processed
            mask_bool = mask.astype(bool)

            with tempfile.TemporaryDirectory() as temp_dir:
                num_tiles_y = (full_window.height + PROCESSING_TILE_SIZE - 1) // PROCESSING_TILE_SIZE
                num_tiles_x = (full_window.width + PROCESSING_TILE_SIZE - 1) // PROCESSING_TILE_SIZE
                logging.info(f"Starting tile processing with {num_tiles_y * num_tiles_x} tiles")

                tasks = []
                for y in range(num_tiles_y):
                    for x in range(num_tiles_x):
                        tile_window = Window(
                            col_off=full_window.col_off + x * PROCESSING_TILE_SIZE,
                            row_off=full_window.row_off + y * PROCESSING_TILE_SIZE,
                            width=min(PROCESSING_TILE_SIZE, full_window.width - x * PROCESSING_TILE_SIZE),
                            height=min(PROCESSING_TILE_SIZE, full_window.height - y * PROCESSING_TILE_SIZE)
                        )
                        tasks.append(((y, x), vrt_path, tile_window, temp_dir))

                full_changes = np.zeros((full_window.height, full_window.width), dtype='uint8')
                full_transitions = np.zeros((256, 256), dtype='uint64')
                full_persistence = np.zeros((256,), dtype='uint64')  # Persistent pixels
                full_initial = np.zeros((256,), dtype='uint64')      # Initial pixels
                failed_tiles = []

                logging.info(f"Processing {len(tasks)} tiles with {MAX_WORKERS} workers...")
                start_time = time.time()
                completed_tiles = 0
                
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(calculate_changes_tile, task) for task in tasks]
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tiles"):
                        try:
                            result = future.result(timeout=300)  # 5 minute timeout per tile
                            completed_tiles += 1
                            
                            logging.info(f"Tile result: {type(result)}, first element: {type(result[0]) if result and len(result) > 0 else 'None'}")
                            
                            if result[0] is not None:
                                changes, transitions, persistence, initial, (tile_y, tile_x), failed = result
                                
                                logging.info(f"Processing tile {tile_y},{tile_x} - failed: {failed}")
                                
                                if not failed:
                                    # Check tile contents before adding
                                    tile_changes_count = np.sum(changes > 0)
                                    tile_max_changes = np.max(changes)
                                    logging.info(f"Tile {tile_y},{tile_x}: {tile_changes_count} pixels with changes, max: {tile_max_changes}")
                                    
                                    # Calculate tile position in full array
                                    y_start = tile_y * PROCESSING_TILE_SIZE
                                    y_end = min(y_start + PROCESSING_TILE_SIZE, full_window.height)
                                    x_start = tile_x * PROCESSING_TILE_SIZE
                                    x_end = min(x_start + PROCESSING_TILE_SIZE, full_window.width)
                                    
                                    # Accumulate results
                                    full_changes[y_start:y_end, x_start:x_end] = changes
                                    full_transitions += transitions
                                    full_persistence += persistence
                                    full_initial += initial
                                else:
                                    failed_tiles.append((tile_y, tile_x))
                            else:
                                logging.warning(f"Tile returned None result - marking as failed")
                                failed_tiles.append((0, 0))  # Placeholder coordinates for failed tile
                            # Log progress every 10 tiles
                            if completed_tiles % 10 == 0:
                                elapsed = time.time() - start_time
                                rate = completed_tiles / elapsed if elapsed > 0 else 0
                                remaining = len(tasks) - completed_tiles
                                eta = remaining / rate if rate > 0 else 0
                                logging.info(f"Processed {completed_tiles}/{len(tasks)} tiles. "
                                           f"Rate: {rate:.1f} tiles/sec. ETA: {eta/60:.1f} min")
                                
                        except Exception as e:
                            logging.error(f"Tile processing failed with error: {str(e)}")
                            failed_tiles.append(("unknown", "unknown"))

                logging.info(f"Tile processing completed. Failed tiles: {len(failed_tiles)}")
                if failed_tiles:
                    logging.warning(f"{len(failed_tiles)} tiles failed to process")
                    failed_tiles_path = os.path.join(grid_output_dir, f'{grid_name}_failed_tiles.npy')
                    np.save(failed_tiles_path, np.array(failed_tiles))

                # Apply the mask to the changes data
                logging.info("Applying polygon mask to data...")
                
                # Debug: Check changes data before masking
                total_pixels = full_changes.size
                pixels_inside_polygon = np.sum(mask_bool)
                changes_before_mask = np.sum(full_changes > 0)
                max_changes_before = np.max(full_changes)
                
                logging.info(f"Before masking - Total pixels: {total_pixels:,}")
                logging.info(f"Before masking - Pixels inside polygon: {pixels_inside_polygon:,} ({100*pixels_inside_polygon/total_pixels:.1f}%)")
                logging.info(f"Before masking - Pixels with changes: {changes_before_mask:,}")
                logging.info(f"Before masking - Max changes: {max_changes_before}")
                
                # DEBUG: Final summary of tile processing
                logging.info(f"TILE PROCESSING SUMMARY:")
                logging.info(f"  Total tiles submitted: {len(tasks)}")
                logging.info(f"  Completed tiles: {completed_tiles}")
                logging.info(f"  Failed tiles: {len(failed_tiles)}")
                logging.info(f"  Success rate: {100*(completed_tiles-len(failed_tiles))/completed_tiles:.1f}%")
                
                if changes_before_mask == 0:
                    logging.error("CRITICAL ISSUE: No changes detected in any tile!")
                    logging.error("This suggests:")
                    logging.error("  1. All tiles failed to process")
                    logging.error("  2. All tiles processed but found no year-to-year changes")  
                    logging.error("  3. Changes were calculated but assembly logic is wrong")
                    logging.error("  4. VRT data has issues (all years identical)")
                    
                    # Additional debugging
                    logging.info(f"full_changes array unique values: {np.unique(full_changes)}")
                    logging.info(f"full_transitions sum: {np.sum(full_transitions)}")
                    logging.info(f"full_persistence sum: {np.sum(full_persistence)}")
                    logging.info(f"full_initial sum: {np.sum(full_initial)}")
                
                # DEBUG: Check mask integrity
                mask_inside_count = np.sum(mask_bool)
                mask_outside_count = np.sum(~mask_bool)
                logging.info(f"Mask check - Inside: {mask_inside_count:,}, Outside: {mask_outside_count:,}, Total: {mask_inside_count + mask_outside_count:,}")
                
                if mask_inside_count == 0:
                    logging.error("CRITICAL: Mask has no inside pixels! This will zero out all data.")
                    logging.error("Polygon might be too small, outside bounds, or mask calculation failed.")
                
                full_changes[~mask_bool] = 255  # Use 255 instead of FILL_VALUE to distinguish from "no changes" (0)
                
                # Debug: Check changes data after masking
                changes_after_mask = np.sum(full_changes > 0)
                max_changes_after = np.max(full_changes)
                
                logging.info(f"After masking - Pixels with changes: {changes_after_mask:,}")
                logging.info(f"After masking - Max changes: {max_changes_after}")
                
                if changes_after_mask == 0:
                    logging.warning("WARNING: No changes detected after masking! This suggests:")
                    logging.warning("  1. No land cover changes occurred within the polygon")
                    logging.warning("  2. The polygon is too small to capture meaningful changes")  
                    logging.warning("  3. There's an issue with the change calculation or polygon masking")
                
                # Store data in Zarr format
                logging.info("Storing data in Zarr format...")
                zarr_path = _store_zarr_data(
                    grid_output_dir, full_changes, full_transitions, 
                    full_persistence, full_initial, src, full_window,
                    bounds, grid_name, failed_tiles, vrt_path, len(polygons), terrai_nom, feature_geometry_wkt
                )
                
                logging.info(f"Zarr file created successfully at: {zarr_path}")
                return zarr_path, grid_output_dir

    except Exception as e:
        logging.critical(f"Error processing grid: {str(e)}", exc_info=True)
        raise


def _store_zarr_data(grid_output_dir, full_changes, full_transitions, full_persistence, 
                    full_initial, src, full_window, bounds, grid_name, failed_tiles, 
                    vrt_path, polygon_count, terrai_nom=None, feature_geometry_wkt=None):
    """Store processed data in Zarr format with proper compression and metadata."""
    
    # Use grid_name as prefix for the Zarr file
    zarr_filename = f"{grid_name}_data.zarr"
    zarr_path = os.path.join(grid_output_dir, zarr_filename)
    logging.info(f"Creating Zarr store at: {zarr_path}")
    
    try:
        root = zarr.open(zarr_path, mode='w')
        logging.info("Zarr store opened successfully")
        
        # Store changes data with proper chunking and compression
        logging.info("Storing changes array...")
        if 'changes' in root:
            logging.warning("The 'changes' array already exists in the Zarr store. Overwriting it.")
            del root['changes']
        changes_array = root.zeros(
            'changes',
            shape=full_changes.shape,
            chunks=(VRT_BLOCK_SIZE, VRT_BLOCK_SIZE),
            dtype='uint8',
        )
        changes_array[:] = full_changes
        logging.info(f"Changes array stored: shape {full_changes.shape}")

        # Store persistence and initial counts
        logging.info("Storing persistence and initial counts...")
        root.zeros('persistence_counts', shape=full_persistence.shape, dtype='uint64')[:] = full_persistence
        root.zeros('initial_counts', shape=full_initial.shape, dtype='uint64')[:] = full_initial
        
        # Add metadata for persistence and initial counts
        persistence_dict = {str(cls): int(count) for cls, count in enumerate(full_persistence) if count > 0}
        initial_dict = {str(cls): int(count) for cls, count in enumerate(full_initial) if count > 0}
        
        # Store transitions data
        logging.info("Storing transition matrix...")
        transitions_array = root.zeros(
            'transition_matrix',  # Changed from 'transitions' to 'transition_matrix' for consistency
            shape=full_transitions.shape,
            chunks=(256, 256),
            dtype='uint64',
        )
        transitions_array[:] = full_transitions

        # Store first and last year
        logging.info("Storing first and last year data...")
        try:
            first_year = src.read(1, window=full_window, fill_value=FILL_VALUE)
            root.zeros(
                'first_year',
                shape=first_year.shape,
                dtype=first_year.dtype,
                chunks=(VRT_BLOCK_SIZE, VRT_BLOCK_SIZE)
            )[:] = first_year
            logging.info("First year data stored successfully")
        except Exception as e:
            logging.error(f"Failed to store first year: {str(e)}")
        
        try:
            last_year = src.read(src.count, window=full_window, fill_value=FILL_VALUE)
            root.zeros(
                'last_year',
                shape=last_year.shape,
                dtype=last_year.dtype,
                chunks=(VRT_BLOCK_SIZE, VRT_BLOCK_SIZE)
            )[:] = last_year
            logging.info("Last year data stored successfully")
        except Exception as e:
            logging.error(f"Failed to store last year: {str(e)}")

        # Store metadata
        logging.info("Storing metadata...")
        window_transform = src.window_transform(full_window)
        min_lon, min_lat, max_lon, max_lat = bounds
        
        metadata = {
            'window_transform': json.dumps(list(window_transform.to_gdal())),
            'crs': str(src.crs),
            'height': full_window.height,
            'width': full_window.width,
            'bounds': list(src.window_bounds(full_window)),
            'lat_range': [float(min_lat), float(max_lat)],
            'lon_range': [float(min_lon), float(max_lon)],
            'grid_name': grid_name,
            'failed_tiles': int(len(failed_tiles)),
            'processing_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'vrt_block_size': int(VRT_BLOCK_SIZE),
            'fill_value': int(FILL_VALUE),
            'polygon_count': int(polygon_count),
            'vrt_path': vrt_path,
            'persistence_by_class': json.dumps(persistence_dict),
            'initial_by_class': json.dumps(initial_dict)
        }
        
        # Add terrai_nom if available
        if terrai_nom:
            metadata['terrai_nom'] = terrai_nom
            
        # Add feature geometry if available
        if feature_geometry_wkt:
            metadata['feature_geometry_wkt'] = feature_geometry_wkt
            
        root.attrs.update(metadata)
        logging.info("Metadata stored successfully")
        
        # Verify the file was created
        if os.path.exists(zarr_path):
            file_size = sum(os.path.getsize(os.path.join(zarr_path, f)) 
                          for f in os.listdir(zarr_path) if os.path.isfile(os.path.join(zarr_path, f)))
            logging.info(f"Zarr file verified: {zarr_path} (approx. {file_size / (1024*1024):.1f} MB)")
        else:
            logging.error(f"Zarr file was not created: {zarr_path}")

        return zarr_path
        
    except Exception as e:
        logging.error(f"Error creating Zarr file: {str(e)}")
        raise
