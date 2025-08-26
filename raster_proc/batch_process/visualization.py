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
    
    # 3. Create changes map (if needed)
    # create_changes_map(root, output_dir, file_prefix)


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
    """Create changes frequency visualization with PNGW for GIS compatibility."""
    grid_name = root.attrs['grid_name']
    changes = root['changes'][:]
    transform = json.loads(root.attrs['window_transform'])
    
    # Create the changes map
    plt.figure(figsize=(12, 20))
    plt.imshow(changes, cmap='gist_stern')
    plt.colorbar(label='Number of Changes (1985-2024)')
    plt.title(f"{grid_name} Change Frequency")
    
    # Add polygon outline if GeoJSON path is provided
    if geojson_path:
        _overlay_polygons(root, plt, geojson_path, changes.shape)
    
    # Save the PNG file
    output_path = os.path.join(output_dir, f'{file_prefix}_changes_map.png')
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()
    
    # Create PNGW file for georeferencing
    _create_pngw_file(output_path, rasterio.transform.Affine(*transform))


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
                y_img = ((np.array(y) - ymin) / (ymax - ymin)) * height
                
                plt.plot(x_img, y_img, color='red', linewidth=3, alpha=0.9, linestyle='-')
                
                # Add interior holes if they exist
                for interior in geometry.interiors:
                    x_hole, y_hole = zip(*interior.coords)
                    x_hole_img = ((np.array(x_hole) - xmin) / (xmax - xmin)) * width
                    y_hole_img = ((np.array(y_hole) - ymin) / (ymax - ymin)) * height
                    plt.plot(x_hole_img, y_hole_img, color='red', linewidth=2, alpha=0.9, linestyle='-')
                    
            elif geometry.geom_type == 'MultiPolygon':
                for poly in geometry.geoms:
                    x, y = zip(*poly.exterior.coords)
                    x_img = ((np.array(x) - xmin) / (xmax - xmin)) * width
                    y_img = ((np.array(y) - ymin) / (ymax - ymin)) * height
                    plt.plot(x_img, y_img, color='red', linewidth=3, alpha=0.9, linestyle='-')
                    
                    # Add interior holes if they exist
                    for interior in poly.interiors:
                        x_hole, y_hole = zip(*interior.coords)
                        x_hole_img = ((np.array(x_hole) - xmin) / (xmax - xmin)) * width
                        y_hole_img = ((np.array(y_hole) - ymin) / (ymax - ymin)) * height
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
                    y_img = ((np.array(y) - ymin) / (ymax - ymin)) * height
                    
                    plt.plot(x_img, y_img, color='red', linewidth=3, alpha=0.9, linestyle='-')
                    
                    # Add interior holes if they exist
                    for interior in geom.interiors:
                        x_hole, y_hole = zip(*interior.coords)
                        x_hole_img = ((np.array(x_hole) - xmin) / (xmax - xmin)) * width
                        y_hole_img = ((np.array(y_hole) - ymin) / (ymax - ymin)) * height
                        plt.plot(x_hole_img, y_hole_img, color='red', linewidth=2, alpha=0.9, linestyle='-')
                        
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = zip(*poly.exterior.coords)
                        x_img = ((np.array(x) - xmin) / (xmax - xmin)) * width
                        y_img = ((np.array(y) - ymin) / (ymax - ymin)) * height
                        plt.plot(x_img, y_img, color='red', linewidth=3, alpha=0.9, linestyle='-')
                        
                        # Add interior holes if they exist
                        for interior in poly.interiors:
                            x_hole, y_hole = zip(*interior.coords)
                            x_hole_img = ((np.array(x_hole) - xmin) / (xmax - xmin)) * width
                            y_hole_img = ((np.array(y_hole) - ymin) / (ymax - ymin)) * height
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
