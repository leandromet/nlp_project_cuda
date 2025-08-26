# -*- coding: utf-8 -*-
"""
Transition visualization module for creating advanced land cover transition visualizations.

This module provides functionality to create treemap and stacked bar visualizations
showing detailed transitions between land cover classes.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import zarr

from config import COLOR_MAP, LABELS


def create_class_transition_visualization(zarr_path, output_dir, vis_type='treemap', 
                                        min_pixels=1000, min_transition_pct=1.0, top_n=10):
    """
    Create visualizations showing where each class transitioned to.
    
    Parameters:
        zarr_path: Path to Zarr file containing the data
        output_dir: Directory to save visualizations
        vis_type: 'treemap' or 'stacked_bar'
        min_pixels: Minimum initial pixels for a class to be included
        min_transition_pct: Minimum percentage of a transition to display (filters out minor transitions)
        top_n: Maximum number of classes to show
    """
    try:
        # Extract prefix from zarr filename for consistent file naming
        zarr_filename = os.path.basename(zarr_path)
        if zarr_filename.endswith('_data.zarr'):
            file_prefix = zarr_filename.replace('_data.zarr', '')
        else:
            file_prefix = 'output'  # Fallback
        
        # Open Zarr dataset
        root = zarr.open(zarr_path, mode='r')
        
        # Check required data
        if 'persistence_counts' not in root or 'initial_counts' not in root:
            logging.error("The Zarr dataset does not contain necessary data arrays for transition visualization")
            return
        
        # Read necessary data
        persistence_counts = np.array(root['persistence_counts'][:])
        initial_counts = np.array(root['initial_counts'][:])
        grid_name = root.attrs.get('grid_name', 'unknown_grid')
        
        # Check if transition matrix exists
        if 'transition_matrix' not in root:
            logging.error("Transition matrix not found in Zarr file. Cannot create transition visualization.")
            return
        
        transition_matrix = np.array(root['transition_matrix'][:])
        
        # Calculate persistence percentages
        persistence_pct = np.zeros_like(persistence_counts, dtype=float)
        for i, (init, pers) in enumerate(zip(initial_counts, persistence_counts)):
            if init > 0:
                persistence_pct[i] = 100.0 * pers / init
        
        # Filter to classes with significant data
        valid_classes = [cls for cls in range(len(initial_counts)) 
                        if (initial_counts[cls] >= min_pixels) and (cls in LABELS)]
        
        if not valid_classes:
            logging.warning(f"No valid classes found with at least {min_pixels} pixels")
            return
        
        # Sort classes by initial count (descending) and limit to top_n
        valid_classes.sort(key=lambda x: -initial_counts[x])
        if len(valid_classes) > top_n:
            valid_classes = valid_classes[:top_n]
        
        # Create visualizations
        if vis_type == 'treemap':
            create_class_transition_treemaps(valid_classes, initial_counts, persistence_counts, 
                                            transition_matrix, grid_name, output_dir, 
                                            min_transition_pct, file_prefix)
        else:
            create_class_transition_stacked_bars(valid_classes, initial_counts, persistence_counts, 
                                                transition_matrix, grid_name, output_dir, 
                                                min_transition_pct, file_prefix)
        
        logging.info(f"Created transition visualization for {file_prefix}")
        
    except Exception as e:
        logging.error(f"Error creating transition visualization: {e}")


def create_class_transition_treemaps(valid_classes, initial_counts, persistence_counts, 
                                    transition_matrix, grid_name, output_dir, 
                                    min_transition_pct, file_prefix):
    """Create treemaps showing transitions for each class."""
    try:
        import squarify
    except ImportError:
        logging.error("squarify package is required for treemap visualization. Install with: pip install squarify")
        return
    
    # Create figure
    fig = plt.figure(figsize=(20, len(valid_classes) * 2.5))
    gs = gridspec.GridSpec(len(valid_classes), 2, width_ratios=[1, 3])
    
    # Process each class
    for i, from_class in enumerate(valid_classes):
        # Get transitions data for this class
        class_transitions = transition_matrix[from_class, :]
        total_pixels = initial_counts[from_class]
        persisted = persistence_counts[from_class]
        changed = total_pixels - persisted
        
        # Create left subplot for bar
        ax_bar = fig.add_subplot(gs[i, 0])
        
        # Create bar showing persistence vs change
        bars = ax_bar.bar([0, 1], [persisted, changed], 
                        color=[COLOR_MAP.get(from_class, '#999999'), '#cccccc'])
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.01 * total_pixels,
                      f"{int(height):,}", ha='center', va='bottom', fontsize=10)
        
        # Add persistence percentage
        pers_pct = 100.0 * persisted / total_pixels if total_pixels > 0 else 0
        ax_bar.set_title(f"{from_class}: {LABELS.get(from_class, '?')} ({pers_pct:.1f}% persistent)")
        ax_bar.set_xticks([0, 1])
        ax_bar.set_xticklabels(['Persistent', 'Changed'])
        ax_bar.set_ylim(0, total_pixels * 1.1)  # Leave room for labels
        
        # Create right subplot for treemap
        ax_treemap = fig.add_subplot(gs[i, 1])
        
        # Get transition destinations, excluding persistence
        destinations = []
        sizes = []
        colors = []
        labels = []
        
        for to_class in range(len(class_transitions)):
            if to_class == from_class:  # Skip persistence
                continue
                
            count = class_transitions[to_class]
            if count > 0:
                pct = 100.0 * count / changed if changed > 0 else 0
                if pct >= min_transition_pct:  # Filter small transitions
                    destinations.append(to_class)
                    sizes.append(count)
                    colors.append(COLOR_MAP.get(to_class, '#999999'))
                    labels.append(f"{to_class}: {LABELS.get(to_class, '?')}\n{count:,} pixels\n{pct:.1f}%")
        
        # Create treemap if we have destinations
        if sizes:
            squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.7, ax=ax_treemap)
            ax_treemap.set_title(f"Transitions from {LABELS.get(from_class, '?')}")
        else:
            ax_treemap.text(0.5, 0.5, "No significant transitions", ha='center', va='center')
            
        ax_treemap.axis('off')
    
    # Overall title
    plt.suptitle(f"{grid_name} Land Cover Transitions Analysis (1985-2023)", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save visualization
    output_path = os.path.join(output_dir, f'{file_prefix}_class_transitions_treemap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved class transitions treemap to {output_path}")


def create_class_transition_stacked_bars(valid_classes, initial_counts, persistence_counts, 
                                        transition_matrix, grid_name, output_dir, 
                                        min_transition_pct, file_prefix):
    """Create stacked bars showing transitions for each class."""
    # Create figure
    fig = plt.figure(figsize=(22, len(valid_classes) * 1.8))
    gs = gridspec.GridSpec(len(valid_classes), 2, width_ratios=[1, 4])
    
    # Process each class
    for i, from_class in enumerate(valid_classes):
        # Get transitions data for this class
        class_transitions = transition_matrix[from_class, :]
        total_pixels = initial_counts[from_class]
        persisted = persistence_counts[from_class]
        changed = total_pixels - persisted
        
        # Create left subplot for class label
        ax_label = fig.add_subplot(gs[i, 0])
        ax_label.text(0.5, 0.5, f"{from_class}: {LABELS.get(from_class, '?')}", 
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(facecolor=COLOR_MAP.get(from_class, '#999999'), alpha=0.7, boxstyle='round'))
        ax_label.axis('off')
        
        # Create right subplot for stacked bars
        ax_bar = fig.add_subplot(gs[i, 1])
        
        # First bar: original total
        ax_bar.bar(0, total_pixels, color=COLOR_MAP.get(from_class, '#999999'), alpha=0.4,
                 label='Original total')
        
        # Second bar: persistent portion
        ax_bar.bar(1, persisted, color=COLOR_MAP.get(from_class, '#999999'),
                 label='Persistent')
        
        # Get transition destinations for third bar
        destinations = []
        counts = []
        colors = []
        labels = []
        bottom = 0
        
        for to_class in range(len(class_transitions)):
            if to_class == from_class:  # Skip persistence
                continue
                
            count = class_transitions[to_class]
            if count > 0:
                pct = 100.0 * count / changed if changed > 0 else 0
                if pct >= min_transition_pct:  # Filter small transitions
                    destinations.append(to_class)
                    counts.append(count)
                    colors.append(COLOR_MAP.get(to_class, '#999999'))
                    labels.append(f"→ {to_class}: {LABELS.get(to_class, '?')} ({pct:.1f}%)")
        
        # Third bar: transitions stacked
        for j, (count, color, label) in enumerate(zip(counts, colors, labels)):
            ax_bar.bar(2, count, bottom=bottom, color=color, label=label)
            
            # Add label if segment is large enough
            if count > 0.05 * total_pixels:  # Only label if >5% of total
                ax_bar.text(2, bottom + count/2, f"{count:,}", ha='center', va='center', 
                           fontsize=8, color='black')
            bottom += count
        
        # Add overall labels
        ax_bar.text(0, total_pixels + 0.01 * total_pixels, 
                  f"Total: {total_pixels:,}", ha='center', va='bottom')
        
        pers_pct = 100.0 * persisted / total_pixels if total_pixels > 0 else 0
        ax_bar.text(1, persisted + 0.01 * total_pixels, 
                  f"Persistent: {persisted:,}\n({pers_pct:.1f}%)", 
                  ha='center', va='bottom')
        
        if changed > 0:
            ax_bar.text(2, bottom + 0.01 * total_pixels, 
                      f"Changed: {changed:,}\n({100-pers_pct:.1f}%)", 
                      ha='center', va='bottom')
        
        # Set axis properties
        ax_bar.set_xticks([0, 1, 2])
        ax_bar.set_xticklabels(['Initial (1985)', 'Persistent', 'Changed by class'])
        ax_bar.set_ylim(0, total_pixels * 1.1)  # Leave room for labels
        ax_bar.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Create a separate legend outside the subplots
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), 
              loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=min(5, len(by_label)))
    
    # Overall title
    plt.suptitle(f"{grid_name} Land Cover Transitions Analysis (1985-2023)", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save visualization
    output_path = os.path.join(output_dir, f'{file_prefix}_class_transitions_stacked.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved class transitions stacked bars to {output_path}")
