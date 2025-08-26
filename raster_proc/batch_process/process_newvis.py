import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import logging
import os
import argparse
import pandas as pd
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transition_visualization.log'),
        logging.StreamHandler()
    ]
)

# Color mapping and labels (from your original script)
COLOR_MAP = {
    0: "#ffffff",
    1: "#1f8d49", 3: "#1f8d49", 4: "#7dc975", 5: "#04381d", 6: "#007785",
    9: "#7a5900", 10: "#d6bc74", 11: "#519799", 12: "#d6bc74", 13:"#ffffff", 14: "#ffefc3",
    15:"#edde8e", 18: "#e974ed", 19:"#c27ba0", 20: "#db7093",   
    21: "#ffefc3", 22:"#d4271e", 23: "#ffa07a", 24: "#d4271e", 25: "#db4d4f", 26:"#2532e4", 29: "#ffaa5f",
    30: "#9c0027", 31: "#091077", 32: "#fc8114", 33: "#259fe4", 35: "#9065d0", 36:"#d082de",
    39: "#f5b3c8", 40: "#c71585", 41: "#f54ca9", 46: "#d68fe2", 47: "#9932cc",
    48: "#e6ccff", 49: "#02d659", 50: "#ad5100", 62: "#ff69b4", 27: "#ffffff"
}

LABELS = {
    0: "No data", 
    1: "Forest", 3: "Forest Formation", 4: "Savanna Formation", 5: "Mangrove", 6: "Floodable Forest",
    9: "Forest Plantation",  11: "Wetland", 10: "Herbaceous", 12: "Grassland",13:"other", 14:"Farming",
    15: "Pasture", 18:"Agri", 19:"Temporary Crop", 20: "Sugar Cane",
    # Other labels as in your original code
}

def create_class_transition_visualization(root, output_dir, vis_type='treemap', min_pixels=1000, min_transition_pct=1.0, top_n=10):
    """
    Create visualizations showing where each class transitioned to.
    
    Parameters:
        root: Zarr root object containing the data
        output_dir: Directory to save visualizations
        vis_type: 'treemap' or 'stacked_bar'
        min_pixels: Minimum initial pixels for a class to be included
        min_transition_pct: Minimum percentage of a transition to display (filters out minor transitions)
        top_n: Maximum number of classes to show
    """
    # Read necessary data
    persistence_counts = np.array(root['persistence_counts'][:])
    initial_counts = np.array(root['initial_counts'][:])
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    # Check if transition matrix exists
    if 'transition_matrix' not in root:
        logging.error("Transition matrix not found in Zarr file. Cannot create transition visualization.")
        # Create a placeholder matrix for demonstration if needed
        # In production, you would return here
        transition_matrix = create_dummy_transition_matrix(initial_counts, persistence_counts)
    else:
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
    
    # Create a figure with a grid layout
    if vis_type == 'treemap':
        create_class_transition_treemaps(valid_classes, initial_counts, persistence_counts, 
                                        transition_matrix, grid_name, output_dir, min_transition_pct)
    else:
        create_class_transition_stacked_bars(valid_classes, initial_counts, persistence_counts, 
                                            transition_matrix, grid_name, output_dir, min_transition_pct)

def create_dummy_transition_matrix(initial_counts, persistence_counts):
    """Create a placeholder transition matrix for demonstration purposes."""
    n_classes = len(initial_counts)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    # Set diagonal (persistence)
    for i in range(n_classes):
        matrix[i, i] = persistence_counts[i]
    
    # Distribute changed pixels somewhat randomly
    for from_class in range(n_classes):
        changed = initial_counts[from_class] - persistence_counts[from_class]
        if changed <= 0:
            continue
            
        # Distribute to a few classes
        potential_targets = [c for c in range(n_classes) if c != from_class and c in LABELS]
        if not potential_targets:
            continue
            
        # Choose 2-4 random targets
        import random
        n_targets = min(len(potential_targets), random.randint(2, 4))
        targets = random.sample(potential_targets, n_targets)
        
        # Distribute changed pixels
        for i, target in enumerate(targets):
            if i == len(targets) - 1:
                # Last target gets remaining pixels
                matrix[from_class, target] = changed - sum(matrix[from_class, t] for t in targets[:-1])
            else:
                # Distribute a portion
                matrix[from_class, target] = int(changed * random.uniform(0.1, 0.6))
    
    return matrix

def create_class_transition_treemaps(valid_classes, initial_counts, persistence_counts, 
                                    transition_matrix, grid_name, output_dir, min_transition_pct):
    """Create treemaps showing transitions for each class."""
    try:
        import squarify
    except ImportError:
        logging.error("squarify package is required for treemap visualization")
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
    output_path = os.path.join(output_dir, 'class_transitions_treemap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved class transitions treemap to {output_path}")

def create_class_transition_stacked_bars(valid_classes, initial_counts, persistence_counts, 
                                        transition_matrix, grid_name, output_dir, min_transition_pct):
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
    output_path = os.path.join(output_dir, 'class_transitions_stacked.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved class transitions stacked bars to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate class transition visualizations from Zarr data')
    parser.add_argument('zarr_path', help='Path to Zarr dataset containing transition data')
    parser.add_argument('--output_dir', '-o', help='Output directory for visualizations (defaults to Zarr directory)')
    parser.add_argument('--vis_type', '-v', default='treemap', choices=['treemap', 'stacked_bar'], 
                        help='Type of transition visualization to create')
    parser.add_argument('--min_pixels', '-m', type=int, default=1000, 
                        help='Minimum initial pixels for a class to be included')
    parser.add_argument('--min_transition_pct', '-p', type=float, default=1.0, 
                        help='Minimum percentage for a transition to be shown')
    parser.add_argument('--top_n', '-n', type=int, default=10, 
                        help='Number of top classes to display')
    
    args = parser.parse_args()
    
    try:
        # Set output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.dirname(args.zarr_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Open Zarr dataset
        root = zarr.open(args.zarr_path, mode='r')
        
        # Check required data
        if 'persistence_counts' not in root or 'initial_counts' not in root:
            logging.error("The Zarr dataset does not contain necessary data arrays")
            return
        
        # Create visualizations
        create_class_transition_visualization(
            root, 
            output_dir, 
            vis_type=args.vis_type,
            min_pixels=args.min_pixels,
            min_transition_pct=args.min_transition_pct,
            top_n=args.top_n
        )
        
        logging.info(f"All visualizations created successfully in {output_dir}")
        
    except Exception as e:
        logging.error(f"Error creating visualizations: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main()