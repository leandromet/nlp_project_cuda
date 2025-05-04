#Usage
# You can use this script from the command line to create visualizations directly from existing Zarr files:

#python visualize_persistence.py /path/to/data.zarr --output_dir /path/to/output --vis_type all

# Command-line Arguments
# zarr_path: Path to the Zarr dataset containing persistence data
# --output_dir or -o: Output directory for visualizations (defaults to same directory as Zarr file)
# --vis_type or -v: Type of visualization to create (options: stacked_bar, pie, horizontal_bar, treemap, grouped_bar, all)
# --min_pixels or -m: Minimum pixels to include a class (default: 1000)
# --top_n or -n: Number of top classes to display (default: 15)
# Visualization Types
# This script offers multiple visualization options:

# Stacked Bar Chart: Shows persistent and changed pixels for each class
# Pie Charts: Compares initial composition to persistent land cover
# Horizontal Bar Chart: Ranks classes by persistence percentage
# Grouped Bar Chart: Compares initial and persistent pixel counts side by side
# Treemap: Shows initial area with persistence percentages (requires the squarify package)
# Example
# To create all visualization types for the top 10 classes with at least 5000 pixels:

#python visualize_persistence.py /path/to/data.zarr -o /path/to/output -v all -m 5000 -n 10

# You can now experiment with different visualization parameters without having to reprocess the original data.




import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from matplotlib.patches import Patch
import logging
import os
import argparse
import pandas as pd
from matplotlib.colors import ListedColormap
import squarify
from matplotlib.sankey import Sankey
            


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('persistence_visualization.log'),
        logging.StreamHandler()
    ]
)

# Color mapping and labels (copied from original script)
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
    21: "Mosaic of Uses", 22:"Non vegetated", 23: "Beach and Sand", 24: "Urban Area",
    25: "Other non Vegetated Areas", 26:"Water", 29: "Rocky Outcrop", 30: "Mining", 31: "Aquaculture",
    32: "Hypersaline Tidal Flat", 33: "River Lake and Ocean", 35: "Palm Oil", 36:"Perennial Crop", 39: "Soybean",
    40: "Rice", 41: "Other Temporary Crops", 46: "Coffee", 47: "Citrus", 48: "Other Perennial Crops",
    49: "Wooded Sandbank Vegetation", 50: "Herbaceous Sandbank Vegetation", 62: "Cotton", 27: "Not Observed"
}

def create_persistence_visualization(root, output_dir, vis_type='stacked_bar', min_pixels=1000, top_n=15):
    """
    Create visualizations showing persistent and changed pixels by class.
    
    Parameters:
        root: Zarr root object
        output_dir: Directory to save visualizations
        vis_type: Type of visualization ('stacked_bar', 'pie', 'horizontal_bar', 'treemap', 'grouped_bar')
        min_pixels: Minimum number of pixels required to include a class
        top_n: Number of top classes to display (by initial count)
    """
    persistence_counts = np.array(root['persistence_counts'][:])
    initial_counts = np.array(root['initial_counts'][:])
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    # Calculate changed counts (initial - persistent)
    changed_counts = initial_counts - persistence_counts
    persistence_pct = np.zeros_like(persistence_counts, dtype=float)
    for i, (init, pers) in enumerate(zip(initial_counts, persistence_counts)):
        if init > 0:
            persistence_pct[i] = 100.0 * pers / init
    
    # Filter to only classes we have labels for and that have significant data
    valid_classes = [cls for cls in range(len(initial_counts)) 
                   if (initial_counts[cls] >= min_pixels) and (cls in LABELS)]
    
    if not valid_classes:
        logging.warning(f"No valid classes found with at least {min_pixels} pixels")
        return
    
    # Sort classes by initial count (descending)
    valid_classes.sort(key=lambda x: -initial_counts[x])
    
    # Limit to top N classes for clarity
    if len(valid_classes) > top_n:
        valid_classes = valid_classes[:top_n]
    
    # Prepare data for plotting
    classes = valid_classes
    labels = [f"{cls}: {LABELS[cls]}" for cls in classes]
    persistent = [persistence_counts[cls] for cls in classes]
    changed = [changed_counts[cls] for cls in classes]
    colors = [COLOR_MAP.get(cls, '#999999') for cls in classes]
    
    # Create DataFrame for easier data manipulation
    df = pd.DataFrame({
        'Class': classes,
        'Label': [LABELS[cls] for cls in classes],
        'Initial': [initial_counts[cls] for cls in classes],
        'Persistent': persistent,
        'Changed': changed,
        'Persistence_Pct': [persistence_pct[cls] for cls in classes],
        'Color': colors
    })
    
    # Create different types of visualizations
    if vis_type == 'stacked_bar':
        create_stacked_bar(df, grid_name, output_dir)
    elif vis_type == 'pie':
        create_pie_charts(df, grid_name, output_dir)
    elif vis_type == 'horizontal_bar':
        create_horizontal_bar(df, grid_name, output_dir)
    elif vis_type == 'treemap':
        create_treemap(df, grid_name, output_dir)
    elif vis_type == 'grouped_bar':
        create_grouped_bar(df, grid_name, output_dir)
    elif vis_type == 'grouped_treemap':
        create_grouped_bar_with_treemap(df, grid_name, output_dir)
    elif vis_type == 'sankey':
        create_sankey_diagram(df, grid_name, output_dir)
    elif vis_type == 'all':
        create_stacked_bar(df, grid_name, output_dir)
        create_pie_charts(df, grid_name, output_dir)
        create_horizontal_bar(df, grid_name, output_dir)
        create_grouped_bar(df, grid_name, output_dir)
        create_grouped_bar_with_treemap(df, grid_name, output_dir)
        create_sankey_diagram(df, grid_name, output_dir)
        try:
            create_treemap(df, grid_name, output_dir)
        except ImportError:
            logging.warning("Treemap visualization requires squarify package. Skipping.")
    else:
        logging.warning(f"Unknown visualization type: {vis_type}")
        create_stacked_bar(df, grid_name, output_dir)
    
    # Always save detailed statistics to CSV
    csv_path = os.path.join(output_dir, 'class_persistence_stats.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved persistence statistics to {csv_path}")

def create_stacked_bar(df, grid_name, output_dir):
    """Create stacked bar chart of persistence/change"""
    plt.figure(figsize=(14, 10))
    
    # Plot changed portion first (bottom)
    bars_changed = plt.bar(df['Label'], df['Changed'], color=df['Color'], 
                          alpha=0.6, label='Changed')
    
    # Plot persistent portion on top
    bars_persistent = plt.bar(df['Label'], df['Persistent'], bottom=df['Changed'], 
                             color=df['Color'], alpha=1.0, label='Persistent')
    
    # Add value labels
    for i, (bar_changed, bar_persistent) in enumerate(zip(bars_changed, bars_persistent)):
        # Only label if there's enough space
        if bar_changed.get_height() > 0.02 * df['Initial'].max():
            plt.text(bar_changed.get_x() + bar_changed.get_width()/2.,
                    bar_changed.get_height()/2.,
                    f"{int(bar_changed.get_height()):,}",
                    ha='center', va='center', color='white', fontsize=8)
        
        if bar_persistent.get_height() > 0.02 * df['Initial'].max():
            plt.text(bar_persistent.get_x() + bar_persistent.get_width()/2.,
                    bar_changed.get_height() + bar_persistent.get_height()/2.,
                    f"{int(bar_persistent.get_height()):,}",
                    ha='center', va='center', color='white', fontsize=8)
            
        # Add percentage label on top
        plt.text(bar_persistent.get_x() + bar_persistent.get_width()/2.,
                bar_changed.get_height() + bar_persistent.get_height() + 0.02 * df['Initial'].max(),
                f"{df['Persistence_Pct'].iloc[i]:.1f}%",
                ha='center', va='bottom', color='black', fontsize=8)
    
    plt.title(f"{grid_name} Land Cover Persistence by Class (1985-2023)")
    plt.ylabel("Number of Pixels")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'class_persistence_stacked.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved stacked bar visualization to {output_path}")

def create_pie_charts(df, grid_name, output_dir):
    """Create pie charts for initial composition and persistence percentages"""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Pie chart of initial land cover composition
    wedges, texts, autotexts = ax1.pie(
        df['Initial'], 
        labels=df['Label'], 
        colors=df['Color'],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        textprops={'fontsize': 10}
    )
    for text in texts:
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_fontsize(8)
    ax1.set_title(f'Initial Land Cover Composition (1985)')
    
    # Pie chart of persistence percentages
    wedges, texts, autotexts = ax2.pie(
        df['Persistent'], 
        labels=df['Label'], 
        colors=df['Color'],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        textprops={'fontsize': 10}
    )
    for text in texts:
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_fontsize(8)
    ax2.set_title(f'Persistent Land Cover (Unchanged 1985-2023)')
    
    plt.suptitle(f"{grid_name} Land Cover Composition and Persistence", fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'class_persistence_pie.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved pie chart visualization to {output_path}")

def create_horizontal_bar(df, grid_name, output_dir):
    """Create horizontal bar chart of persistence percentages"""
    # Sort by persistence percentage
    df_sorted = df.sort_values(by='Persistence_Pct', ascending=False).copy()
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(df_sorted['Label'], df_sorted['Persistence_Pct'], color=df_sorted['Color'])
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(min(width + 1, 101), 
                bar.get_y() + bar.get_height()/2, 
                f"{width:.1f}%", 
                va='center', 
                fontsize=8)
    
    plt.xlim(0, 105)  # Leave room for labels
    plt.title(f"{grid_name} Land Cover Persistence Percentage (1985-2023)")
    plt.xlabel("Persistence (%)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'class_persistence_horizontal.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved horizontal bar visualization to {output_path}")

def create_grouped_bar(df, grid_name, output_dir):
    """Create grouped bar chart comparing initial and persistent pixels"""
    # Prepare data for grouped bar chart
    x = np.arange(len(df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 10))
    bar1 = ax.bar(x - width/2, df['Initial'], width, label='Initial (1985)', color=[c+'99' for c in df['Color']])
    bar2 = ax.bar(x + width/2, df['Persistent'], width, label='Persistent', color=df['Color'])
    
    # Add percentage labels
    for i, (b1, b2) in enumerate(zip(bar1, bar2)):
        persistence_pct = df['Persistence_Pct'].iloc[i]
        ax.text(i, b2.get_height() + 0.01 * df['Initial'].max(),
               f"{persistence_pct:.1f}%",
               ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Land Cover Class')
    ax.set_ylabel('Number of Pixels')
    ax.set_title(f"{grid_name} Initial vs. Persistent Land Cover (1985-2023)")
    ax.set_xticks(x)
    ax.set_xticklabels(df['Label'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'class_persistence_grouped.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved grouped bar visualization to {output_path}")

def create_grouped_bar_with_treemap(df, grid_name, output_dir):
    """Create grouped bar chart with treemap showing transitions of changed pixels"""
    try:
        import squarify
    except ImportError:
        logging.error("squarify package is required for treemap visualization")
        return
    
    # Create figure with two subplots - bar chart on left, treemap on right
    fig = plt.figure(figsize=(20, 12))
    
    # Left subplot for grouped bars (initial and persistent)
    ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
    
    # Right subplot for treemap (where changed pixels went)
    ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=3)
    
    # Prepare data for grouped bar chart
    x = np.arange(len(df))
    width = 0.35
    
    # Plot grouped bars on left subplot
    bar1 = ax1.bar(x - width/2, df['Initial'], width, label='Initial (1985)', color=[c+'99' for c in df['Color']])
    bar2 = ax1.bar(x + width/2, df['Persistent'], width, label='Persistent', color=df['Color'])
    
    # Add percentage labels on bars
    for i, (b1, b2) in enumerate(zip(bar1, bar2)):
        persistence_pct = df['Persistence_Pct'].iloc[i]
        ax1.text(i, b2.get_height() + 0.01 * df['Initial'].max(),
               f"{persistence_pct:.1f}%",
               ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Land Cover Class')
    ax1.set_ylabel('Number of Pixels')
    ax1.set_title('Initial vs. Persistent Land Cover')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Label'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create treemap of changed pixels on right subplot
    plt.sca(ax2)
    
    # Calculate changed pixels for each class
    changed_values = df['Initial'] - df['Persistent']
    
    # Create informative labels for treemap boxes
    treemap_labels = [f"{row['Label']}\n{changed_values.iloc[i]:,} pixels\n{100*changed_values.iloc[i]/changed_values.sum():.1f}% of change" 
                     for i, (_, row) in enumerate(df.iterrows()) if changed_values.iloc[i] > 0]
    treemap_sizes = [val for val in changed_values if val > 0]
    treemap_colors = [color for color, changed in zip(df['Color'], changed_values) if changed > 0]
    
    # If we have changed pixels to show
    if len(treemap_sizes) > 0:
        squarify.plot(sizes=treemap_sizes, 
                     label=treemap_labels, 
                     color=treemap_colors, 
                     alpha=0.8, 
                     ax=ax2,
                     text_kwargs={'fontsize': 11})
        ax2.set_title('Distribution of Changed Pixels by Original Class')
        ax2.axis('off')
        
        # Add explanatory note
        plt.figtext(0.65, 0.05, 
                    "Treemap shows distribution of pixels that changed from 1985-2023.\n" +
                    "Box size represents number of pixels that changed from that class.",
                    ha='center', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    else:
        ax2.text(0.5, 0.5, "No significant changes detected", 
                ha='center', va='center', fontsize=14)
        ax2.axis('off')
    
    # Overall title
    plt.suptitle(f"{grid_name} Land Cover Change Analysis (1985-2023)", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the plot
    output_path = os.path.join(output_dir, 'class_persistence_grouped_treemap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved grouped bar with treemap to {output_path}")

def create_sankey_diagram(df, grid_name, output_dir):
    """Create a static Sankey diagram for persistence and change flows"""

    # Prepare data for Sankey diagram
    labels = df['Label'].tolist()
    initial = df['Initial'].tolist()
    persistent = df['Persistent'].tolist()
    changed = df['Changed'].tolist()

    # Create flows and labels for Sankey
    flows = []
    sankey_labels = []
    colors = []

    for i, label in enumerate(labels):
        # Add flow for persistent pixels
        flows.append(persistent[i])
        sankey_labels.append(f"{label} (Persistent)")
        colors.append(df['Color'].iloc[i])

        # Add flow for changed pixels
        flows.append(-changed[i])
        sankey_labels.append(f"{label} (Changed)")
        colors.append(df['Color'].iloc[i])

    # Create Sankey diagram
    fig, ax = plt.subplots(figsize=(14, 10))
    sankey = Sankey(ax=ax, unit=None)

    for i in range(0, len(flows), 2):
        sankey.add(flows=[flows[i], flows[i + 1]],
                    labels=[sankey_labels[i], sankey_labels[i + 1]],
                    orientations=[0, 0],
                    facecolor=colors[i // 2])

    sankey.finish()
    ax.set_title(f"{grid_name} Land Cover Persistence and Change Flows (1985-2023)")

    # Save the plot
    output_path = os.path.join(output_dir, 'class_persistence_sankey.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved Sankey diagram visualization to {output_path}")

    

    
def create_treemap(df, grid_name, output_dir):
    """Create treemap visualization for persistence statistics"""
    
    
    # Create a larger figure for the treemap
    plt.figure(figsize=(16, 12))
    
    # Calculate the size values and normalize for better visualization
    sizes = df['Initial'].values
    # Calculate labels with both class name and persistence percentage
    labels = [f"{row['Label']}\n{row['Persistence_Pct']:.1f}%" for _, row in df.iterrows()]
    
    # Create treemap
    squarify.plot(sizes=sizes, label=labels, color=df['Color'], alpha=0.8, text_kwargs={'fontsize':12})
    plt.axis('off')
    plt.title(f"{grid_name} Land Cover Initial Areas with Persistence % (1985-2023)", fontsize=16)
    
    # Add a legend explaining the visualization
    plt.figtext(0.5, 0.02, 
                "Box size represents initial area in 1985. Percentage shows proportion that remained unchanged through 2023.",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Save the plot
    output_path = os.path.join(output_dir, 'class_persistence_treemap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved treemap visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate persistence visualizations from Zarr data')
    parser.add_argument('zarr_path', help='Path to Zarr dataset containing persistence data')
    parser.add_argument('--output_dir', '-o', help='Output directory for visualizations (defaults to Zarr directory)')
    parser.add_argument('--vis_type', '-v', default='all', choices=['stacked_bar', 'pie', 'horizontal_bar', 'treemap', 'grouped_bar', 'grouped_treemap', 'sankey', 'all'], 
                        help='Type of visualization to create')
    parser.add_argument('--min_pixels', '-m', type=int, default=1000, help='Minimum pixels to include a class')
    parser.add_argument('--top_n', '-n', type=int, default=15, help='Number of top classes to display')
    
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
        
        # Check if persistence data exists
        if 'persistence_counts' not in root or 'initial_counts' not in root:
            logging.error("The Zarr dataset does not contain persistence_counts or initial_counts arrays")
            return
        
        # Create visualizations
        create_persistence_visualization(
            root, 
            output_dir, 
            vis_type=args.vis_type,
            min_pixels=args.min_pixels,
            top_n=args.top_n
        )
        
        logging.info(f"All visualizations created successfully in {output_dir}")
        
    except Exception as e:
        logging.error(f"Error creating visualizations: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main()