# -*- coding: utf-8 -*-
"""
Sankey diagram module for visualizing land cover transitions.

Creates interactive and static Sankey diagrams showing transitions
between land cover classes.
"""

import os
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

from config import COLOR_MAP, LABELS


def create_sankey_diagram(zarr_path, output_dir):
    """Create Sankey diagram showing land cover transitions."""
    import zarr
    
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Check if transition matrix exists
        if 'transition_matrix' not in root:
            logging.error(f"No transition matrix found in {zarr_path}. Cannot create Sankey diagram.")
            return
            
        transition_matrix = np.array(root['transition_matrix'][:])
        grid_name = root.attrs.get('grid_name', 'unknown_grid')
        
        # Extract prefix from zarr filename instead of grid_name
        zarr_filename = os.path.basename(zarr_path)
        if zarr_filename.endswith('_data.zarr'):
            file_prefix = zarr_filename.replace('_data.zarr', '')
        else:
            file_prefix = grid_name  # Fallback to grid_name
        
        logging.info(f"Creating Sankey diagram with prefix: {file_prefix}")
        logging.info(f"Transition matrix shape: {transition_matrix.shape}")
        logging.info(f"Total transitions: {np.sum(transition_matrix):,}")
        
        # Filter matrix to significant transitions
        filtered_matrix, source_labels, target_labels = _filter_transition_matrix(
            transition_matrix, min_pixels=50  # Reduced threshold for more inclusive diagrams
        )
        
        if filtered_matrix is None:
            logging.warning("No significant transitions found for Sankey diagram")
            return
        
        logging.info(f"Filtered matrix has {len(source_labels)} classes with {np.sum(filtered_matrix):,} total transitions")
        
        # Create Sankey diagram
        fig = _create_sankey_figure(
            filtered_matrix, source_labels, target_labels, grid_name
        )
        
        # Save as HTML
        html_path = os.path.join(output_dir, f'{file_prefix}_sankey_diagram.html')
        plot(fig, filename=html_path, auto_open=False)
        
        # Also save as static PNG
        try:
            png_path = os.path.join(output_dir, f'{file_prefix}_sankey_diagram.png')
            fig.write_image(png_path, width=1200, height=600, scale=2)
            logging.info(f"Saved Sankey diagram: {html_path} and {png_path}")
        except Exception as e:
            logging.warning(f"Could not save PNG version of Sankey diagram: {e}")
            logging.info(f"Saved Sankey diagram: {html_path}")
            
    except Exception as e:
        logging.error(f"Error creating Sankey diagram for {zarr_path}: {e}")
        logging.exception("Full traceback:")
    html_path = os.path.join(output_dir, f'{file_prefix}_sankey_transitions.html')
    plot(fig, filename=html_path, auto_open=False)
    
    # Save transition matrix as CSV
    _save_transition_csv(filtered_matrix, source_labels, target_labels, output_dir, file_prefix)
    
    logging.info(f"Created Sankey diagram at {html_path}")


def _filter_transition_matrix(transition_matrix, min_pixels=100):
    """Filter transition matrix to show only significant transitions."""
    # Get classes that have significant data
    significant_classes = []
    for i in range(transition_matrix.shape[0]):
        if (np.sum(transition_matrix[i, :]) >= min_pixels or 
            np.sum(transition_matrix[:, i]) >= min_pixels):
            if i in LABELS:  # Only include classes we have labels for
                significant_classes.append(i)
    
    if len(significant_classes) < 2:
        return None, None, None
    
    # Filter matrix to significant classes
    indices = significant_classes
    filtered_matrix = transition_matrix[np.ix_(indices, indices)]
    
    # Create labels for source and target
    source_labels = [f"{cls}: {LABELS[cls]} (1985)" for cls in indices]
    target_labels = [f"{cls}: {LABELS[cls]} (2024)" for cls in indices]
    
    return filtered_matrix, source_labels, target_labels


def _create_sankey_figure(matrix, source_labels, target_labels, grid_name):
    """Create the Sankey diagram figure."""
    n_classes = len(source_labels)
    
    # Prepare data for Sankey diagram
    source_indices = []
    target_indices = []
    values = []
    
    for i in range(n_classes):
        for j in range(n_classes):
            if matrix[i, j] > 0:  # Only include non-zero transitions
                source_indices.append(i)
                target_indices.append(j + n_classes)  # Offset target indices
                values.append(matrix[i, j])
    
    # Create node colors
    node_colors = []
    class_indices = list(range(n_classes))
    
    # Source nodes (1985)
    for i in class_indices:
        class_num = _extract_class_number(source_labels[i])
        color = COLOR_MAP.get(class_num, '#999999')
        node_colors.append(color)
    
    # Target nodes (2024)
    for i in class_indices:
        class_num = _extract_class_number(target_labels[i])
        color = COLOR_MAP.get(class_num, '#999999')
        node_colors.append(color)
    
    # Create link colors (same as source)
    link_colors = []
    for source_idx in source_indices:
        class_num = _extract_class_number(source_labels[source_idx])
        color = COLOR_MAP.get(class_num, '#999999')
        # Add transparency to links
        if color.startswith('#'):
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            link_colors.append(f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.6)')
        else:
            link_colors.append(color)
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=source_labels + target_labels,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title_text=f"{grid_name} Land Cover Transitions (1985 → 2024)",
        font_size=12,
        height=600,
        width=1200
    )
    
    return fig


def _extract_class_number(label):
    """Extract class number from label string."""
    try:
        return int(label.split(':')[0])
    except (ValueError, IndexError):
        return 0


def _save_transition_csv(matrix, source_labels, target_labels, output_dir, file_prefix):
    """Save transition matrix as CSV file."""
    # Create DataFrame
    df = pd.DataFrame(
        matrix,
        index=[label.split(' (1985)')[0] for label in source_labels],
        columns=[label.split(' (2024)')[0] for label in target_labels]
    )
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f'{file_prefix}_transition_matrix.csv')
    df.to_csv(csv_path)
    
    # Also save summary statistics
    summary_path = os.path.join(output_dir, f'{file_prefix}_transition_summary.csv')
    with open(summary_path, 'w') as f:
        f.write("Source_Class,Target_Class,Pixels,Percentage_of_Source\n")
        
        for i, source_label in enumerate(source_labels):
            source_total = np.sum(matrix[i, :])
            for j, target_label in enumerate(target_labels):
                if matrix[i, j] > 0:
                    percentage = 100 * matrix[i, j] / source_total if source_total > 0 else 0
                    source_clean = source_label.split(' (1985)')[0]
                    target_clean = target_label.split(' (2024)')[0]
                    f.write(f'"{source_clean}","{target_clean}",'
                           f'{matrix[i, j]},{percentage:.2f}%\n')


def create_simple_sankey_html(zarr_path, output_dir):
    """Create a simplified HTML Sankey diagram for quick viewing."""
    import zarr
    
    root = zarr.open(zarr_path, mode='r')
    transition_matrix = np.array(root['transition_matrix'][:])
    grid_name = root.attrs.get('grid_name', 'unknown_grid')
    
    # Get top transitions only
    filtered_matrix, source_labels, target_labels = _filter_transition_matrix(
        transition_matrix, min_pixels=500  # Higher threshold for simplicity
    )
    
    if filtered_matrix is None:
        logging.warning("No significant transitions for simple Sankey")
        return
    
    # Create basic HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{grid_name} Land Cover Transitions</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div id="sankey" style="width:100%;height:600px;"></div>
        <script>
            // Sankey data would be embedded here
            var data = [{_create_plotly_data_dict(filtered_matrix, source_labels, target_labels)}];
            var layout = {{
                title: '{grid_name} Land Cover Transitions (1985 → 2024)',
                font: {{size: 12}}
            }};
            Plotly.newPlot('sankey', data, layout);
        </script>
    </body>
    </html>
    """
    
    simple_html_path = os.path.join(output_dir, 'sankey_simple.html')
    with open(simple_html_path, 'w') as f:
        f.write(html_content)


def _create_plotly_data_dict(matrix, source_labels, target_labels):
    """Create Plotly data dictionary for embedding in HTML."""
    # This is a simplified version for demonstration
    # In practice, you'd want to properly format the data
    return """
    type: "sankey",
    node: {
        label: ["Forest", "Agriculture", "Urban", "Water"],
        color: ["#228B22", "#DEB887", "#696969", "#4169E1"]
    },
    link: {
        source: [0, 0, 1, 2],
        target: [1, 2, 3, 3],
        value: [100, 50, 25, 10]
    }
    """
