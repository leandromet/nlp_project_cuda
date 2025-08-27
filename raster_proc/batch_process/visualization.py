# -*- coding: utf-8 -*-
"""
Visualization module for creating maps and charts from processed data.

Handles creation of land cover maps, persistence visualizations,
and change frequency maps.
"""

import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import geopandas as gpd
import rasterio

from config import COLOR_MAP, LABELS


def create_visualizations(zarr_path, output_dir, geojson_path=None):
    """Create all visualizations from the Zarr data."""
    import zarr
    
    root = zarr.open(zarr_path, mode='r')
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    # Extract prefix from zarr filename instead of grid_name
    zarr_filename = os.path.basename(zarr_path)
    if zarr_filename.endswith('_data.zarr'):
        file_prefix = zarr_filename.replace('_data.zarr', '')
    else:
        file_prefix = grid_name  # Fallback to grid_name
    
    logging.info(f"Creating visualizations with prefix: {file_prefix}")
    
    # 1. Create land cover maps with polygon overlay
    create_landcover_map(root, output_dir, file_prefix, geojson_path)
    create_initial_landcover_map(root, output_dir, file_prefix, geojson_path)
    
    # 2. Create persistence visualization
    create_persistence_visualization(root, output_dir, file_prefix)
    
    # 3. Create changes map showing total changes per pixel over 40 years
    create_changes_map(root, output_dir, file_prefix, geojson_path)


def create_landcover_map(root, output_dir, file_prefix, geojson_path=None):
    """Create land cover visualization with PNGW for GIS compatibility."""
    
    grid_name = root.attrs['grid_name']
    last_year = root['last_year'][:]
    transform = rasterio.transform.Affine(*json.loads(root.attrs['window_transform']))
    
    # Create colormap
    cmap = ListedColormap([COLOR_MAP.get(i, '#ffffff') for i in range(max(COLOR_MAP.keys()) + 1)])
    
    plt.figure(figsize=(12, 20))
    plt.imshow(last_year, cmap=cmap, vmin=0, vmax=max(COLOR_MAP.keys()))
    plt.title(f"{grid_name} Land Cover")
    
    # Add polygon outline if GeoJSON path is provided
    if geojson_path:
        _overlay_polygons(root, plt, geojson_path, last_year.shape)
    
    # Create legend
    _create_legend(plt)
    
    # Overlay the polygon name if available
    if 'UNIDADES_H' in root.attrs:
        UNIDADES_H = root.attrs['UNIDADES_H']
        plt.text(
            0.5, 0.95, UNIDADES_H, 
            fontsize=16, color='white', ha='center', va='center', 
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.5')
        )
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{file_prefix}_landcover_map.png')
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()
    
    # Create PNGW file for georeferencing
    _create_pngw_file(output_path, transform)


def create_initial_landcover_map(root, output_dir, file_prefix, geojson_path=None):
    """Create initial land cover visualization with PNGW for GIS compatibility."""
    
    grid_name = root.attrs['grid_name']
    first_year = root['first_year'][:]
    transform = rasterio.transform.Affine(*json.loads(root.attrs['window_transform']))
    
    # Create colormap
    cmap = ListedColormap([COLOR_MAP.get(i, '#ffffff') for i in range(max(COLOR_MAP.keys()) + 1)])
    
    plt.figure(figsize=(12, 20))
    plt.imshow(first_year, cmap=cmap, vmin=0, vmax=max(COLOR_MAP.keys()))
    plt.title(f"{grid_name} Initial Land Cover (1985)")
    
    # Add polygon outline if GeoJSON path is provided
    if geojson_path:
        _overlay_polygons(root, plt, geojson_path, first_year.shape)
    
    # Create legend
    _create_legend(plt)
    
    # Overlay the polygon name if available
    if 'UNIDADES_H' in root.attrs:
        UNIDADES_H = root.attrs['UNIDADES_H']
        plt.text(
            0.5, 0.95, UNIDADES_H, 
            fontsize=16, color='white', ha='center', va='center', 
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.5')
        )
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{file_prefix}_initial_landcover_map.png')
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()
    
    # Create PNGW file for georeferencing
    _create_pngw_file(output_path, transform)


def create_persistence_visualization(root, output_dir, file_prefix):
    """Create stacked bar chart showing persistent and changed pixels by class."""
    persistence_counts = np.array(root['persistence_counts'][:])
    initial_counts = np.array(root['initial_counts'][:])
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    # Calculate changed counts (initial - persistent)
    changed_counts = initial_counts - persistence_counts
    
    # Filter to only classes we have labels for and that have data
    valid_classes = [cls for cls in range(len(initial_counts)) 
                   if (initial_counts[cls] > 0) and (cls in LABELS)]
    
    if not valid_classes:
        logging.warning("No valid classes found for persistence visualization")
        return
    
    # Sort classes by initial count (descending)
    valid_classes.sort(key=lambda x: -initial_counts[x])
    
    # Prepare data for plotting
    classes = valid_classes
    labels = [f"{cls}: {LABELS[cls]}" for cls in classes]
    persistent = [persistence_counts[cls] for cls in classes]
    changed = [changed_counts[cls] for cls in classes]
    colors = [COLOR_MAP.get(cls, '#999999') for cls in classes]
    
    # Create stacked bar chart
    plt.figure(figsize=(14, 10))
    
    # Plot changed portion first (bottom)
    bars_changed = plt.bar(labels, changed, color=colors, 
                          alpha=0.6, label='Changed')
    
    # Plot persistent portion on top
    bars_persistent = plt.bar(labels, persistent, bottom=changed, 
                             color=colors, alpha=1.0, label='Persistent')
    
    # Add value labels
    for bar_changed, bar_persistent in zip(bars_changed, bars_persistent):
        # Only label if there's enough space
        if bar_changed.get_height() > 0.1 * max(initial_counts):
            plt.text(bar_changed.get_x() + bar_changed.get_width()/2.,
                    bar_changed.get_height()/2.,
                    f"{int(bar_changed.get_height()):,}",
                    ha='center', va='center', color='white', fontsize=8)
        
        if bar_persistent.get_height() > 0.1 * max(initial_counts):
            plt.text(bar_persistent.get_x() + bar_persistent.get_width()/2.,
                    bar_changed.get_height() + bar_persistent.get_height()/2.,
                    f"{int(bar_persistent.get_height()):,}",
                    ha='center', va='center', color='white', fontsize=8)
    
    plt.title(f"{grid_name} Land Cover Persistence by Class (1985-2024)")
    plt.ylabel("Number of Pixels")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{file_prefix}_class_persistence_stacked.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed statistics to CSV
    csv_path = os.path.join(output_dir, f'{file_prefix}_class_persistence_stacked.csv')
    with open(csv_path, 'w') as f:
        f.write("Class,Label,Color,Initial_Pixels,Persistent_Pixels,Changed_Pixels,Persistent_Percent\n")
        for cls in classes:
            persistent_pct = 100 * persistence_counts[cls] / initial_counts[cls] if initial_counts[cls] > 0 else 0
            f.write(f"{cls},{LABELS[cls]},{COLOR_MAP.get(cls, '#FFFFFF')},"
                   f"{initial_counts[cls]},{persistence_counts[cls]},"
                   f"{changed_counts[cls]},{persistent_pct:.1f}%\n")
    
    logging.info(f"Saved stacked persistence visualization to {output_path}")


def create_changes_map(root, output_dir, file_prefix, geojson_path=None):
    """Create changes frequency visualization showing total changes per pixel over 40 years with PNGW for GIS compatibility."""
    
    # Check if changes data exists
    if 'changes' not in root:
        logging.error("No changes data found in Zarr file. Cannot create changes map.")
        return
        
    grid_name = root.attrs['grid_name']
    changes = root['changes'][:]
    transform = rasterio.transform.Affine(*json.loads(root.attrs['window_transform']))
    
    logging.info(f"Creating changes map for {grid_name}")
    logging.info(f"Changes array shape: {changes.shape}, dtype: {changes.dtype}")
    logging.info(f"Changes range: {np.min(changes)} to {np.max(changes)}")
    logging.info(f"Changes unique values: {np.unique(changes)}")
    logging.info(f"Changes value distribution: {np.bincount(changes.flatten())}")
    
    # Debug: Check if all values are really 0
    non_zero_count = np.sum(changes != 0)
    logging.info(f"DEBUG: Non-zero pixels in changes array: {non_zero_count:,}")
    if non_zero_count > 0:
        logging.info(f"DEBUG: Max non-zero value: {np.max(changes[changes != 0])}")
        logging.info(f"DEBUG: Sample of non-zero values: {changes[changes != 0][:10]}")
    else:
        logging.warning("DEBUG: ALL CHANGES VALUES ARE ZERO! This indicates a problem.")
    
    # Mask out fill values (0 usually indicates no data or outside polygon)
    # We need to distinguish between "no changes" and "outside analysis area"
    # Check if we can identify the analysis area from other data
    valid_data_mask = np.ones(changes.shape, dtype=bool)
    
    # Try to get a better mask from the first/last year data to identify analysis area
    if 'first_year' in root and 'last_year' in root:
        first_year = root['first_year'][:]
        last_year = root['last_year'][:]
        # Pixels that have valid land cover data (not fill value) in both first and last year
        # should be considered part of the analysis area
        from config import FILL_VALUE
        valid_data_mask = (first_year != FILL_VALUE) & (last_year != FILL_VALUE)
        logging.info(f"Using first/last year data to identify analysis area: {np.sum(valid_data_mask):,} valid pixels")
    else:
        # Fallback: use changes array to identify analysis area
        # Areas outside polygon should have been set to 255, inside polygon has 0 or more changes
        valid_data_mask = (changes != 255)  # 255 is used for outside polygon areas
        logging.info(f"Using changes array to identify analysis area: {np.sum(valid_data_mask):,} valid pixels")
        if np.sum(valid_data_mask) == 0:
            logging.warning("No valid analysis area found using changes array mask!")
    
    # Calculate statistics only for the valid analysis area
    total_analysis_pixels = np.sum(valid_data_mask)
    changes_in_analysis_area = changes[valid_data_mask]
    
    if total_analysis_pixels == 0:
        logging.error("No valid analysis area found!")
        return
    
    # Remove any remaining fill values (255) from the analysis area calculations
    # This can happen if the mask isn't perfect
    changes_in_analysis_area = changes_in_analysis_area[changes_in_analysis_area != 255]
    actual_analysis_pixels = len(changes_in_analysis_area)  # Actual count after removing fill values
    
    changed_pixels = np.sum(changes_in_analysis_area > 0)
    stable_pixels = np.sum(changes_in_analysis_area == 0)
    max_changes = np.max(changes_in_analysis_area)
    avg_changes = np.mean(changes_in_analysis_area[changes_in_analysis_area > 0]) if changed_pixels > 0 else 0
    
    logging.info(f"Analysis area statistics:")
    logging.info(f"  Total analysis pixels: {actual_analysis_pixels:,}")
    logging.info(f"  Changed pixels: {changed_pixels:,} ({100*changed_pixels/actual_analysis_pixels:.1f}%)")
    logging.info(f"  Stable pixels: {stable_pixels:,} ({100*stable_pixels/actual_analysis_pixels:.1f}%)")
    logging.info(f"  Max changes: {max_changes}")
    logging.info(f"  Note: Color scale will use range 0 to {max_changes} for better contrast")
    
    if changed_pixels == 0:
        logging.warning("No land cover changes detected in the analysis area!")
        logging.info("This might indicate:")
        logging.info("  1. The area is very stable (unlikely for 40 years)")
        logging.info("  2. There's an issue with change calculation")
        logging.info("  3. The polygon mask is incorrectly applied")
    
    # Create the changes map
    plt.figure(figsize=(12, 20))
    
    # Use a more informative colormap for changes
    # Set vmin=0 but only show meaningful changes, excluding fill values
    if max_changes == 0:
        logging.warning("Maximum changes is 0 - creating map anyway but data might be incorrect")
        max_changes = 1  # Avoid division by zero
    
    # Create a masked array to show only the analysis area
    changes_display = changes.copy().astype(float)
    changes_display[~valid_data_mask] = np.nan  # Set areas outside analysis to NaN
    changes_display[changes_display == 255] = np.nan  # Also mask the 255 fill values
    
    # Use a better color scale that focuses on actual changes (0 to max_changes, not 0 to 255)
    # This will give much better contrast for the actual change values
    im = plt.imshow(changes_display, cmap='plasma', vmin=0, vmax=max_changes)
    cbar = plt.colorbar(im, label='Number of Land Cover Transitions (1985-2024)', shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    
    plt.title(f"{grid_name} Land Cover Change Frequency\n(Total transitions per pixel over 40 years)", fontsize=14)
    plt.axis('off')  # Remove axes for cleaner look
    
    # Add polygon outline if GeoJSON path is provided
    if geojson_path:
        _overlay_polygons(root, plt, geojson_path, changes.shape)
    
    # Calculate statistics for pixels within the valid area
    stats_text = f"""Change Analysis (within study area):
    • Analysis area: {actual_analysis_pixels:,} pixels
    • Pixels with land cover changes: {changed_pixels:,} ({100*changed_pixels/actual_analysis_pixels:.1f}%)
    • Stable pixels (no changes): {stable_pixels:,} ({100*stable_pixels/actual_analysis_pixels:.1f}%)
    • Maximum transitions per pixel: {max_changes}
    • Average transitions (changed pixels): {avg_changes:.1f}
    
    Time period: 40 years (1985-2024)
    Each unit = one land cover class transition"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the PNG file
    output_path = os.path.join(output_dir, f'{file_prefix}_changes_map.png')
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()
    
    # Create PNGW file for georeferencing
    _create_pngw_file(output_path, transform)
    
    logging.info(f"Saved changes frequency map to {output_path}")
    logging.info(f"Change statistics - Analysis area: {actual_analysis_pixels:,}, Changed: {changed_pixels:,} ({100*changed_pixels/actual_analysis_pixels:.1f}%), Max transitions: {max_changes}")
    
    # Save detailed change statistics to CSV
    csv_path = os.path.join(output_dir, f'{file_prefix}_change_statistics.csv')
    with open(csv_path, 'w') as f:
        f.write("Metric,Value,Percentage\n")
        f.write(f"Analysis area pixels,{actual_analysis_pixels},100.0%\n")
        f.write(f"Stable pixels,{stable_pixels},{100*stable_pixels/actual_analysis_pixels:.1f}%\n")
        f.write(f"Changed pixels,{changed_pixels},{100*changed_pixels/actual_analysis_pixels:.1f}%\n")
        f.write(f"Maximum transitions,{max_changes},\n")
        f.write(f"Average transitions (changed pixels),{avg_changes:.2f},\n")
        
        # Add distribution of change frequencies
        if changed_pixels > 0:
            f.write("\nChange Frequency,Pixel Count,Percentage of Analysis Area\n")
            change_counts = np.bincount(changes_in_analysis_area.flatten())
            for i, count in enumerate(change_counts):
                if count > 0:
                    f.write(f"{i} transitions,{count},{100*count/actual_analysis_pixels:.2f}%\n")
    
    logging.info(f"Saved detailed change statistics to {csv_path}")


def _overlay_polygons(root, plt, geojson_path, img_shape):
    """Overlay polygons from stored geometry in Zarr metadata onto the map."""
    try:
        # First try to use stored geometry from Zarr metadata
        if 'feature_geometry_wkt' in root.attrs:
            from shapely import wkt
            geometry = wkt.loads(root.attrs['feature_geometry_wkt'])
            
            # Get bounds from the root attributes
            bounds = json.loads(root.attrs['bounds']) if isinstance(root.attrs['bounds'], str) else root.attrs['bounds']
            xmin, ymin, xmax, ymax = bounds
            height, width = img_shape
            
            logging.info(f"Overlaying stored polygon geometry on map. Bounds: {bounds}, Image shape: {img_shape}")
            
            # Draw the stored geometry
            if geometry.geom_type == 'Polygon':
                # Convert coordinates to image space
                x, y = zip(*geometry.exterior.coords)
                x_img = ((np.array(x) - xmin) / (xmax - xmin)) * width
                # Flip Y-axis: image coordinates have Y=0 at top, geographic has Y=max at top
                y_img = height - ((np.array(y) - ymin) / (ymax - ymin)) * height
                
                plt.plot(x_img, y_img, color='red', linewidth=3, alpha=0.9, linestyle='-')
                
                # Add interior holes if they exist
                for interior in geometry.interiors:
                    x_hole, y_hole = zip(*interior.coords)
                    x_hole_img = ((np.array(x_hole) - xmin) / (xmax - xmin)) * width
                    # Flip Y-axis for holes too
                    y_hole_img = height - ((np.array(y_hole) - ymin) / (ymax - ymin)) * height
                    plt.plot(x_hole_img, y_hole_img, color='red', linewidth=2, alpha=0.9, linestyle='-')
                    
            elif geometry.geom_type == 'MultiPolygon':
                for poly in geometry.geoms:
                    x, y = zip(*poly.exterior.coords)
                    x_img = ((np.array(x) - xmin) / (xmax - xmin)) * width
                    # Flip Y-axis for MultiPolygon too
                    y_img = height - ((np.array(y) - ymin) / (ymax - ymin)) * height
                    plt.plot(x_img, y_img, color='red', linewidth=3, alpha=0.9, linestyle='-')
                    
                    # Add interior holes if they exist
                    for interior in poly.interiors:
                        x_hole, y_hole = zip(*interior.coords)
                        x_hole_img = ((np.array(x_hole) - xmin) / (xmax - xmin)) * width
                        # Flip Y-axis for holes in MultiPolygon
                        y_hole_img = height - ((np.array(y_hole) - ymin) / (ymax - ymin)) * height
                        plt.plot(x_hole_img, y_hole_img, color='red', linewidth=2, alpha=0.9, linestyle='-')
            
            logging.info(f"Successfully overlaid stored polygon geometry on the map")
            return
            
    except Exception as e:
        logging.warning(f"Could not overlay stored polygon geometry: {str(e)}")
        logging.info("Falling back to reading from GeoJSON file...")
    
    # Fallback: try to read from GeoJSON file if geometry not stored or failed
    try:
        if geojson_path and os.path.exists(geojson_path):
            gdf = gpd.read_file(geojson_path)
            if not gdf.empty and 'geometry' in gdf.columns:
                # Get bounds from the root attributes
                bounds = json.loads(root.attrs['bounds']) if isinstance(root.attrs['bounds'], str) else root.attrs['bounds']
                xmin, ymin, xmax, ymax = bounds
                height, width = img_shape
                
                logging.info(f"Overlaying polygon from GeoJSON on map. Bounds: {bounds}, Image shape: {img_shape}")
                
                # Only overlay the first feature (for individual processing)
                geom = gdf.iloc[0].geometry
                if geom.geom_type == 'Polygon':
                    # Convert coordinates to image space
                    x, y = zip(*geom.exterior.coords)
                    x_img = ((np.array(x) - xmin) / (xmax - xmin)) * width
                    # Flip Y-axis: image coordinates have Y=0 at top, geographic has Y=max at top
                    y_img = height - ((np.array(y) - ymin) / (ymax - ymin)) * height
                    
                    plt.plot(x_img, y_img, color='red', linewidth=3, alpha=0.9, linestyle='-')
                    
                    # Add interior holes if they exist
                    for interior in geom.interiors:
                        x_hole, y_hole = zip(*interior.coords)
                        x_hole_img = ((np.array(x_hole) - xmin) / (xmax - xmin)) * width
                        # Flip Y-axis for holes too
                        y_hole_img = height - ((np.array(y_hole) - ymin) / (ymax - ymin)) * height
                        plt.plot(x_hole_img, y_hole_img, color='red', linewidth=2, alpha=0.9, linestyle='-')
                        
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = zip(*poly.exterior.coords)
                        x_img = ((np.array(x) - xmin) / (xmax - xmin)) * width
                        # Flip Y-axis for MultiPolygon in GeoJSON fallback
                        y_img = height - ((np.array(y) - ymin) / (ymax - ymin)) * height
                        plt.plot(x_img, y_img, color='red', linewidth=3, alpha=0.9, linestyle='-')
                        
                        # Add interior holes if they exist
                        for interior in poly.interiors:
                            x_hole, y_hole = zip(*interior.coords)
                            x_hole_img = ((np.array(x_hole) - xmin) / (xmax - xmin)) * width
                            # Flip Y-axis for holes in MultiPolygon fallback
                            y_hole_img = height - ((np.array(y_hole) - ymin) / (ymax - ymin)) * height
                            plt.plot(x_hole_img, y_hole_img, color='red', linewidth=2, alpha=0.9, linestyle='-')
                
                logging.info(f"Successfully overlaid polygon from GeoJSON file")
                
    except Exception as e:
        logging.warning(f"Could not overlay polygon from GeoJSON: {str(e)}")
        logging.exception("Full traceback for polygon overlay error:")


def _create_legend(plt):
    """Create a legend for the land cover map."""
    patches = [Patch(color=COLOR_MAP[k], label=f"{k}: {LABELS[k]}") 
               for k in sorted(LABELS.keys()) if k in COLOR_MAP]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')


def _create_pngw_file(png_path, transform):
    """Create PNGW file for georeferencing."""
    pngw_path = png_path + 'w'
    with open(pngw_path, 'w') as pngw_file:
        pngw_file.write(f"{transform.a}\n")  # Pixel size in x-direction
        pngw_file.write(f"{transform.b}\n")  # Rotation (usually 0)
        pngw_file.write(f"{transform.d}\n")  # Rotation (usually 0)
        pngw_file.write(f"{transform.e}\n")  # Pixel size in y-direction (negative)
        pngw_file.write(f"{transform.c}\n")  # X-coordinate of the upper-left corner
        pngw_file.write(f"{transform.f}\n")  # Y-coordinate of the upper-left corner
