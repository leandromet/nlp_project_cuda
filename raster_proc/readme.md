# Brazil Amazon and Atlantic Forest Land Cover Analysis Documentation

## Overview

The brazil_color_table_amazon.py script provides a robust framework for extracting, analyzing, and visualizing land cover change in the Amazon region of Brazil using MapBiomas classification data from 1985-2023. The script employs parallel processing, memory optimization, and sophisticated visualization techniques to handle large geospatial datasets efficiently. For the Atlantic forest we just need to change coordinates, as this script can be run for any region in Brazil.

![Amazon Land Cover Example] (image-5.png)

![Atlantic forest Land Cover Example] (image-4.png)

## Features

- **Robust Data Extraction**: Implements retry logic and error handling for reliable extraction from large VRT datasets
- **Parallel Processing**: Uses Python's `ProcessPoolExecutor` for efficient multi-core processing
- **Memory Management**: Dynamically monitors and manages memory usage to prevent overflow
- **Chunked Processing**: Processes data in configurable tiles to handle arbitrarily large regions
- **Comprehensive Visualization**:
  - Land cover maps with proper geographic referencing
  - Change frequency heat maps
  - Interactive Sankey diagrams showing land cover transitions
  - Decadal transition analysis
  - Color-coded transition matrices
- **GIS Integration**: Creates PNGW world files for geospatial software compatibility

## Prerequisites

- Python 3.7+
- Required packages:
  ```
  rasterio
  numpy
  zarr
  matplotlib
  plotly
  tqdm
  psutil
  kaleido
  ```
- MapBiomas Collection data as a VRT file

careful, this script can be very resource hungry!! Try smaller windows first (1x1 degree), I managed to handle 8x8 but not 10x10 degrees with 20 cores and 90GB RAM plus a RTX4060 8GB.

![Resource usage during an 8x8 run](<Screenshot From 2025-04-21 22-07-02.png>)

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TILE_SIZE` | 256 | Size of processing tiles in pixels |
| `MAX_WORKERS` | 10 | Maximum number of parallel workers |
| `MEMORY_BUFFER_GB` | 5 | Memory buffer to maintain in GB |
| `MAX_READ_RETRIES` | 3 | Maximum retries for failed reads |
| `RETRY_DELAY` | 0.05 | Delay between retries in seconds |
| `VRT_BLOCK_SIZE` | 512 | Block size matching VRT structure |

## Land Cover Classification

The script uses the MapBiomas classification system with 50+ land cover classes, including:

- Forest formations (classes 1, 3)
- Savanna formation (class 4)
- Mangrove (class 5)
- Floodable forest (class 6)
- Wetlands (class 11)
- Grassland (class 12)
- Pasture (class 15)
- Agriculture (classes 20, 39, 40, 41, etc.)
- Urban areas (class 24)
- Water bodies (class 33)

Each class has a predefined color and label for visualization purposes.

```python
# Color mapping and labels 
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

```

## Core Functions

### Data Processing

#### `robust_read(src, band, window, retries=MAX_READ_RETRIES)`

Reads raster data with retry logic for fault tolerance.

```python
# Example usage
data = robust_read(src, 1, window)
```

#### `process_tile(task)`

Processes a single tile to calculate land cover changes and transitions.

```python
# Task structure
task = (y, x, path, window, temp_dir)
changes, transitions, (y, x) = process_tile(task)
```

#### `extract_grid_data_optimized(vrt_path, polygon_coords, output_dir)`

Extracts and processes data for a specific geographic region.

```python
# Example call for a specific region
zarr_path, grid_output_dir = extract_grid_data_optimized(
    '/path/to/mapbiomas.vrt',
    [((-55, -2), (-55, -7), (-51, -7), (-51, -2), (-55, -2))],
    'output_directory'
)
```

### Visualization

#### `create_landcover_map(root, output_dir)`

Creates a land cover map with proper georeferencing.

#### `create_changes_map(root, output_dir)`

Creates a change frequency map showing where land cover has changed most frequently.
![Amazon change count map](image-6.png)
![Atlantic forest](image-7.png)

#### `create_sankey_diagrams(root, output_dir)`

Creates Sankey diagrams showing land cover transitions across different time periods.


![Amazon forest sankey](image-9.png)
![Atlantic forest sankey](image-8.png)


#### `create_decadal_sankey_diagrams(root, output_dir)`

Creates Sankey diagrams specifically focused on decade-by-decade transitions.

## Output Files

For each analyzed region, the script generates:

### Data Files
- `data.zarr/` - Zarr dataset containing:
  - Original yearly land cover data
  - Change frequency map
  - Transition matrix
  - First year (1985) and last year (2023) snapshots
  - Geographic metadata

### Visualizations
- `landcover_map.png` - Land cover map for 2023
- `landcover_map.pngw` - World file for GIS georeferencing
- `changes_map.png` - Heat map showing frequency of land cover changes
- `changes_map.pngw` - World file for GIS georeferencing

### Transition Analysis
- `transitions_full.html` - Interactive Sankey diagram for entire period
- `transitions_full.csv` - CSV data of all transitions
- `transitions_1985_1995.html` - Interactive Sankey for 1985-1995
- `transitions_1995_2005.html` - Interactive Sankey for 1995-2005
- `transitions_2005_2015.html` - Interactive Sankey for 2005-2015
- `transitions_2015_2023.html` - Interactive Sankey for 2015-2023
- `change_matrix_{start}_{end}.png` - Graphical table of transitions
- `change_matrix_{start}_{end}.csv` - CSV data of transitions

#### Atlantic Forest change matrix
![Atlantic forest change matrix 1985-1995](../change_matrix_1985_1995.png)
![Atlantic forest change matrix 1985-2023](../change_matrix_1985_2023.png)
![Atlantic forest change matrix 2015-2023](../change_matrix_2015_2023.png)

#### Amazon Forest change matrix
![Amazon forest change matrix 1985-1995](<../change_matrix_1985_1995 (2).png>)
![Amazon forest change matrix 1985-2023](<../change_matrix_1985_2023 (2).png>)
![Amazon forest change matrix 2015-2023](<../change_matrix_2015_2023 (2).png>)


## Error Handling

The script implements several error handling mechanisms:

1. **Retry Logic**: Failed read operations are retried with exponential backoff
2. **Window Validation**: Ensures requested windows are within raster bounds
3. **Worker Management**: Tracks and reports failed tiles without stopping the process
4. **Memory Monitoring**: Adjusts processing based on available system memory
5. **Data Validation**: Checks for valid data and handles exceptions appropriately

## Performance Considerations

- Processing an 8×8 degree region (~880×880 km) requires approximately 32GB RAM
- Full processing including visualizations takes 3-5 hours on a modern system
- The script dynamically adjusts to available system resources
- Parallel processing reduces time but increases memory usage
- VRT-specific optimizations improve performance with large datasets

## Usage

```python
# Define the VRT file path
VRT_FILE = '/path/to/mapbiomas_coverage_1985_2023.vrt'

# Define the output directory
OUTPUT_BASE_DIR = 'amazon_analysis_results'

# Define the analysis region (coordinates in lon, lat format)
AMAZON_REGION = [((-55, -2), (-55, -7), (-51, -7), (-51, -2), (-55, -2))]

# Run the analysis
zarr_path, grid_output_dir = extract_grid_data_optimized(
    VRT_FILE, AMAZON_REGION, OUTPUT_BASE_DIR
)

# Create visualizations
create_visualizations(zarr_path, grid_output_dir)
```

## Example Analysis: Central Amazon Region (2° to 7°S, 51° to 55°W)

This region encompasses parts of the states of Pará and Amazonas, including sections of the Tapajós National Forest and areas affected by the BR-163 highway development corridor. The analysis reveals:

1. **Forest Loss Patterns**: Deforestation fronts advancing from the southeast
2. **Transition Hotspots**: Primary forest to pasture conversion dominant near highways
3. **Temporal Patterns**: Acceleration of land cover change after 2005
4. **Land Use Intensification**: Agricultural expansion in previously cleared areas
5. **Protected Area Effectiveness**: Lower change frequencies within protected boundaries

## Troubleshooting

- **Memory Errors**: Reduce `MAX_WORKERS` or increase tile size
- **Failed Tiles**: Check the log for specific error messages
- **VRT Issues**: Ensure the VRT file correctly references all source TIFFs
- **Visualization Errors**: If Sankey diagrams fail, check for missing class colors

---

*Created by Leandro Biondo, UBCO - Sustainability, 2025*



# Brazil Color Table Amazon Analysis Documentation

## 1. Introduction and Setup

This script performs analysis of land cover changes in the Amazon region of Brazil using MapBiomas data. It processes multi-year raster data, identifies land cover transitions, and creates visualizations of the changes.

### Key Dependencies

```python
import rasterio
import numpy as np
import zarr
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import matplotlib.pyplot as plt
import plotly.graph_objects as go
```

### Configuration Constants

```python
# Memory and performance tuning parameters
TILE_SIZE = 256               # Size of processing tiles in pixels
MAX_WORKERS = 10              # Maximum number of parallel worker processes
MEMORY_BUFFER_GB = 5          # Memory buffer to maintain in GB
MAX_READ_RETRIES = 3          # Maximum retries for failed reads
RETRY_DELAY = 0.05            # Delay between retries in seconds (exponential backoff)
```

## 2. MapBiomas Classification System

The script uses the official MapBiomas land cover classification system with 50+ classes. Each class has an assigned color and label for visualization.

```python
# Color mapping (simplified example)
COLOR_MAP = {
    0: "#ffffff",                 # No data
    1: "#1f8d49", 3: "#1f8d49",   # Forest types
    4: "#7dc975",                 # Savanna 
    15: "#edde8e",                # Pasture
    24: "#d4271e",                # Urban Area
    33: "#259fe4"                 # Water bodies
    # ... many other classes ...
}

# Class labels
LABELS = {
    0: "No data", 
    1: "Forest", 3: "Forest Formation", 
    4: "Savanna Formation",
    # ... many other classes ...
}
```

## 3. Robust Read Functionality

The script implements fault-tolerant reading of geospatial data with retries and validation.

```python
def robust_read(src, band, window, retries=MAX_READ_RETRIES):
    """Attempt to read a tile with retries on failure."""
    for attempt in range(retries):
        try:
            # Validate window is within bounds
            valid_window = window.intersection(Window(0, 0, src.width, src.height))
            if valid_window is None or valid_window.width <= 0 or valid_window.height <= 0:
                raise ValueError("Requested window is outside the raster bounds")
                
            # Read data
            data = src.read(band, window=valid_window)
            
            # Validate data
            if np.all(data == src.nodata):
                raise ValueError("All values are nodata")
                
            return data
        except Exception as e:
            # Retry with exponential backoff
            if attempt == retries - 1:
                raise
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    raise RuntimeError(f"Failed after {retries} attempts")
```

## 4. Tile Processing

The script processes large raster data in manageable tiles using parallel workers:

```python
def process_tile(task):
    """Process a single tile for changes and transitions."""
    y, x, path, window, temp_dir = task
    
    try:
        with rasterio.open(path) as src:
            changes = np.zeros((window.height, window.width), dtype='uint8')
            transitions = np.zeros((256, 256), dtype='uint64')
            
            # Read first band/year
            prev_data = src.read(1, window=window, fill_value=FILL_VALUE)
            
            # Process subsequent bands/years
            for band_idx in range(2, src.count + 1):
                curr_data = src.read(band_idx, window=window, fill_value=FILL_VALUE)
                
                # Identify changed pixels
                changed = prev_data != curr_data
                changes += changed
                
                # Record transitions
                if np.any(changed):
                    from_vals = prev_data[changed]
                    to_vals = curr_data[changed]
                    for f, t in zip(from_vals, to_vals):
                        transitions[f, t] += 1
                
                prev_data = curr_data
                
            return changes, transitions, (y, x)
            
    except Exception as e:
        logging.warning(f"Tile {y},{x} failed: {str(e)}")
        return None
```

## 5. Optimized Data Extraction

The main data extraction function processes a geographic region by dividing it into manageable tiles:

```python
def extract_grid_data_optimized(vrt_path, polygon_coords, output_dir):
    """Optimized extraction with VRT-aware settings."""
    # VRT-specific configuration
    VRT_BLOCK_SIZE = 512  # Matches the BlockXSize/BlockYSize in VRT
    TILE_SIZE = 2048      # Multiple of VRT_BLOCK_SIZE for optimal performance
    
    # Important: Getting window aligned to VRT blocks improves performance
    def align_to_blocks(value, block_size, max_value, align_type='floor'):
        if align_type == 'floor':
            return max(0, (value // block_size) * block_size)
        else:
            return min(max_value, ((value + block_size - 1) // block_size) * block_size)
    
    # Calculate window based on geographic coordinates
    # ... coordinate conversion code ...
    
    # Process data in parallel using tiles
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(process_tile, task))
        
        # Process results as they complete
        for future in progress:
            result = future.result()
            if result is not None:
                changes, transitions, (y, x) = result
                # ... store results ...
            else:
                failed_tiles.append((y, x))
    
    # Store data in Zarr format with compression
    zarr_path = os.path.join(grid_output_dir, 'data.zarr')
    root = zarr.open(zarr_path, mode='w')
    
    # Important: Chunking matters for performance
    root.zeros(
        'changes',
        shape=full_changes.shape,
        chunks=(VRT_BLOCK_SIZE, VRT_BLOCK_SIZE),




# Iguaçu National Park Land Cover Analysis

![alt text](image.png)

## Overview

This Python script analyzes land cover changes in the Iguaçu National Park region of Brazil from 1985 to 2023 using MapBiomas Collection data. The script extracts, processes, and visualizes land cover data to reveal patterns of landscape change over nearly four decades, with a focus on transitions between different land cover classes.

## Features

- **Geographic Extraction**: Extracts data specifically for the Iguaçu region (approximately 25.7°S-24.7°S, 54.7°W-53.7°W)
- **Multi-temporal Analysis**: Processes land cover data across all available years (1985-2023)
- **Change Detection**: Identifies and quantifies locations of land cover change
- **Transition Analysis**: Tracks conversions between different land cover types
- **Interactive Visualizations**:
  - Land cover maps with geographic coordinates
  - Change frequency heat maps
  - Sankey diagrams showing land cover transitions by decade
- **Landmarks Identification**: Labels key geographic features (Iguaçu Falls, cities, park boundaries)

## Prerequisites

- Python 3.7+
- Required packages:
  ```
  rasterio
  numpy
  zarr
  matplotlib
  plotly
  tqdm
  ```
- MapBiomas Brazil Collection data as a VRT file

## Data Requirements

The script requires a Virtual Raster (VRT) file that references yearly land cover classification GeoTIFFs from the MapBiomas project:

- **Source**: MapBiomas Collection (recommended Collection 8 or later)
- **Resolution**: 30m
- **Projection**: EPSG:4326 (WGS84)
- **Time Range**: 1985-2023
- **Classification**: MapBiomas land cover classes

## Implementation Details

### Data Extraction

The script extracts data for the Iguaçu region using these steps:

1. Defines target coordinates for the Iguaçu region
2. Converts geographic coordinates (lat/lon) to pixel coordinates
3. Creates a raster window for the specified region
4. Extracts raster data for all available years
5. Calculates pixel-based changes between consecutive years
6. Stores results in a Zarr dataset for efficient access

### Land Cover Classification

The analysis uses the MapBiomas classification system, which includes:

- Forest formations (classes 1, 3)
- Savanna formation (class 4)
- Wetlands (class 11)
- Grasslands (class 12)
- Pasture (class 15)
- Agriculture (classes 20, 39, 40, 41, etc.)
- Urban areas (class 24)
- Water bodies (class 33)
- And many more classes

### Change Analysis

For each pixel, the script:
1. Counts how many times the land cover class changed from 1985 to 2023
2. Identifies the specific transitions between classes
3. Creates visualization of change frequency
4. Analyzes statistical patterns of change

### Sankey Diagram Generation

The script creates Sankey diagrams for four time periods:
- 1985-1995
- 1995-2005
- 2005-2015
- 2015-2023

Each diagram shows:
- Persistence (areas that maintained the same class)
- Transitions (areas that changed from one class to another)
- Proportional flow representation based on pixel counts
- Color-coded flows matching the standard MapBiomas palette

## Usage

1. Set the path to your MapBiomas VRT file:
   ```python
   VRT_FILE = '/path/to/mapbiomas_coverage_1985_2023.vrt'
   ```

2. Set the output directory:
   ```python
   OUTPUT_DIR = 'iguacu_results'
   ```

3. Run the script:
   ```bash
   python iguacu_sankey.py
   ```

## Outputs

The script generates the following outputs in the specified directory:

### Data Files
- `data.zarr/` - Zarr dataset containing:
  - Original land cover data for all years
  - Change frequency map
  - Transition matrix
  - Geographic metadata

### Visualizations
- `extraction_preview.png` - Quick preview of the extracted area
- `iguacu_2023.png` - Land cover map for 2023 with geographic coordinates
- `iguacu_changes.png` - Heat map showing frequency of land cover changes

### Transition Diagrams
- `transitions_1985_1995.html` - Interactive Sankey diagram (HTML)
- `transitions_1985_1995.png` - Static image of Sankey diagram
- `transitions_1995_2005.html/.png` - Diagrams for 1995-2005
- `transitions_2005_2015.html/.png` - Diagrams for 2005-2015
- `transitions_2015_2023.html/.png` - Diagrams for 2015-2023

## Geographic Parameters

- **Target Region**: Iguaçu National Park and surroundings
- **Bounding Box**:
  - Latitude: -25.7°S to -24.7°S (South to North)
  - Longitude: -54.7°W to -53.7°W (West to East)
- **Region Size**: Approximately 100 × 100 km

## Key Landmarks

The script includes the following landmarks (when within the extracted region):

- **Iguaçu National Park**: Primary conservation area (-25.5°S, -54.0°W)
- **Foz do Iguaçu City**: Brazilian city (-25.516°S, -54.588°W)
- **Cataratas do Iguaçu**: The famous waterfalls (-25.695°S, -54.436°W)
- **Puerto Iguazú City**: Argentine city (-25.599°S, -54.573°W)
- **Ciudad del Este**: Paraguayan city (-25.509°S, -54.611°W)

## Notes

- Processing the full dataset requires approximately 2-3GB of RAM
- The script automatically filters out small, insignificant transitions (less than 0.1% of total)
- Visualization includes annotations with statistics about change patterns
- All visualizations include proper geographic coordinates in EPSG:4326

---

Created by Leandro Meneguelli Biondo for the UBCO PhD project, 2025.






# MapBiomas Brazil Grid Analysis

## Overview

This Python script analyzes land cover change patterns across Brazil using MapBiomas data from 1985-2023. The analysis divides Brazil into 5-degree grid cells and processes each cell to identify land cover changes, transitions between classes, and persistence patterns.

![alt text](image-1.png)

## Features

- **Systematic Grid Analysis**: Divides Brazil into 43 grid cells of 5×5 degrees for manageable processing
- **Change Detection**: Quantifies frequency of land cover changes for each pixel from 1985-2023
- **Transition Analysis**: Tracks conversions between different land cover classes (e.g., Forest → Pasture)
- **Optimized Processing**: Uses chunked data processing via Zarr for handling large raster datasets
- **Rich Visualizations**:
  - Land cover maps with geographic coordinates
  - Change frequency heat maps
  - Interactive Sankey diagrams showing land cover flows 

## Prerequisites

- Python 3.7+
- Required packages: 
  ```
  rasterio
  numpy
  zarr
  matplotlib
  plotly
  tqdm
  shapely
  ```
- MapBiomas Brazil Collection 8 data as a VRT file

## Data Source

This script processes the MapBiomas Collection 8 dataset, which provides annual land cover classifications for Brazil from 1985 to 2023 at 30m resolution. The data contains 39 land cover classes including forest formations, savannas, grasslands, pasture, agriculture, and urban areas.

- **VRT File**: The script expects a Virtual Raster (VRT) file that references all yearly land cover GeoTIFFs
- **Coordinate System**: EPSG:4326 (WGS84)
- **Data Structure**: Each band in the VRT represents one year (1985-2023)

## Grid System

The script divides Brazil into 43 cells using a 5-degree grid system:

```
((-74, 6), (-69, 6), (-69, 1), (-74, 1), (-74, 6)),  # Northwest Amazon
((-69, 6), (-64, 6), (-64, 1), (-69, 1), (-69, 6)),  # North Amazon
...
((-54, -29), (-49, -29), (-49, -34), (-54, -34), (-54, -29))  # South Brazil
```

Each cell is processed independently, allowing for parallel processing and manageable memory usage.

## Analysis Process

![alt text](image-2.png)

For each grid cell, the script:

1. **Extracts Data**: Clips the VRT file to the grid cell coordinates
2. **Calculates Changes**: Compares each consecutive year to identify pixels that changed
3. **Builds Transition Matrix**: Tracks all transitions between land cover classes
4. **Stores Results**: Creates a Zarr dataset with original data, changes, and transitions
5. **Generates Visualizations**: Creates maps and diagrams to visualize the results

## Output Files

For each grid cell, the script generates:

- **Zarr Dataset** (`data.zarr`): Contains:
  - Original yearly land cover data
  - Change frequency map
  - Transition matrix
  - Metadata (coordinate system, bounds, etc.)

- **Visualizations**:
  - `extraction_preview.png`: Quick preview of the grid cell
  - `{grid_name}_2023.png`: Map of 2023 land cover
  - `{grid_name}_changes.png`: Map of change frequency
  - `transitions_{start}_{end}.html`: Interactive Sankey diagrams for each decade
  - `transitions_{start}_{end}.png`: Static image of Sankey diagrams

## Key Functions

### `extract_grid_data(vrt_path, polygon_coords, output_dir)`
Extracts and processes data for a single grid cell. Converts geographic coordinates to pixel coordinates, reads data from the VRT file, and calculates changes and transitions.

### `create_decadal_sankey_diagrams(root, output_dir)`
Creates Sankey diagrams showing land cover transitions for four periods: 1985-1995, 1995-2005, 2005-2015, and 2015-2023. Filters out insignificant transitions (less than 0.1% of total).

### `visualize_results(zarr_path, output_dir)`
Generates visualizations from the processed data, including land cover maps and change frequency maps with proper geographic coordinates.

### `get_grid_name(polygon_coords)`
Generates a standardized name for each grid cell based on its coordinates (e.g., `grid_6n1n_74w69w`).

## Usage

1. Set the path to your MapBiomas VRT file in the script:
   ```python
   VRT_FILE = '/path/to/mapbiomas_coverage_1985_2023.vrt'
   ```

2. Set the output directory:
   ```python
   OUTPUT_BASE_DIR = 'grid_results'
   ```

3. Run the script:
   ```bash
   python brazil_grid5.py
   ```

4. Results will be saved in the output directory, with a subdirectory for each grid cell.

## Notes and Limitations

- **Memory Usage**: Processing each 5-degree cell requires approximately 2-4GB of RAM
- **Disk Space**: Each grid cell generates 500MB-1GB of data
- **Processing Time**: Each grid cell takes 10-30 minutes to process on a modern system
- **Edge Effects**: The grid system may introduce edge effects at cell boundaries
- **Data Quality**: The analysis inherits any classification errors from the MapBiomas dataset

---

Created by Leandro Meneguelli Biondo for the UBCO PhD project, 2025.







# Brazil Land Cover Analysis Combiner

## Overview

This Python script combines and analyzes land cover data across Brazil from the MapBiomas dataset (1985-2023). It merges multiple Zarr datasets from grid-based processing into a comprehensive national dataset while implementing memory-efficient techniques for handling very large geospatial data.


![alt text](image-3.png)

## Features

- **Memory-Efficient Data Combination**: Merges multiple Zarr grid datasets into a single comprehensive dataset
- **Chunked Processing**: Uses Zarr's chunked array capabilities to handle datasets larger than available RAM
- **Downsampling**: Implements visualization downsampling for displaying continent-scale data
- **Statistical Sampling**: Uses representative sampling for creating Sankey diagrams of nationwide land cover transitions
- **Comprehensive Metadata**: Preserves geographic information and transformations from source datasets

## Prerequisites

- Python 3.7+
- Required packages:
  ```
  zarr
  numpy
  matplotlib
  plotly
  tqdm
  rasterio
  ```
- Pre-processed MapBiomas grid datasets in Zarr format

## Data Sources

This script expects pre-processed MapBiomas data in Zarr format, typically generated by the `brazil_grid5.py` script that divides Brazil into 5-degree grid cells. Each grid dataset contains:

- Original land cover data (yearly, 1985-2023)
- Change frequency maps
- Transition matrices
- Geographic metadata (bounds, transforms, etc.)

## Core Functions

### `combine_zarr_datasets(input_base_dir, output_dir, max_chunk_size=512)`

Combines multiple grid-based Zarr datasets into a single comprehensive dataset:

1. **Collects Metadata**: Scans all grid datasets to determine combined dimensions and bounds
2. **Creates Output Structure**: Initializes a new Zarr store with appropriate chunking
3. **Combines Data**: Merges data from each grid, preserving geographic relationships
4. **Memory-Efficient Processing**: Uses chunked reads/writes to manage memory usage
5. **Preserves Metadata**: Combines and stores critical geographic information

### `create_full_sankey_diagrams(zarr_path, output_dir, sample_fraction=0.01)`

Creates Sankey diagrams showing land cover transitions between decades:

1. **Random Sampling**: Selects a representative sample (default 1%) of pixels to analyze
2. **Transition Analysis**: Identifies persistent areas and transitions between land cover types
3. **Statistical Scaling**: Scales sample counts to represent the full dataset
4. **Filtering**: Removes statistically insignificant transitions (<0.1% of total)
5. **Interactive Visualization**: Creates interactive Sankey diagrams using Plotly

### `visualize_full_results(zarr_path, output_dir, downsample_factor=10)`

Generates visualizations for the combined dataset:

1. **Downsampling**: Reduces resolution by a specified factor (default 10x) for visualization
2. **Geographic Rendering**: Maintains proper geographic coordinates and bounds
3. **Legend Creation**: Includes comprehensive legend with all land cover classes
4. **High-Quality Export**: Produces publication-quality images with appropriate metadata

## Usage

1. Set input and output directories in the script:
   ```python
   INPUT_BASE_DIR = '/path/to/mapbiomas_proc_zarr'  # Directory containing grid Zarr datasets
   OUTPUT_DIR = '/path/to/combined_results'         # Output directory for combined results
   ```

2. Run the script:
   ```bash
   python brazil_analyse_combine.py
   ```

3. The script will:
   - Combine all grid datasets into a single Zarr store
   - Create downsampled visualizations of the entire dataset
   - Generate Sankey diagrams for nationwide land cover transitions

## Output Files

- **Combined Zarr Dataset**: `combined_data.zarr` containing:
  - Full land cover data for all years
  - Nationwide change frequency map
  - Comprehensive transition matrix
  - Combined metadata

- **Visualizations**:
  - `downsampled_10_full_dataset_2023.png`: Land cover map of Brazil for 2023 (10x downsampled)
  - `downsampled_10_full_dataset_changes.png`: Change frequency map (10x downsampled)

- **Sankey Diagrams**:
  - `sampled_transitions_1985_1995.html/.png`: Interactive/static diagrams showing transitions
  - `sampled_transitions_1995_2005.html/.png`: for each decadal period
  - `sampled_transitions_2005_2015.html/.png`:
  - `sampled_transitions_2015_2023.html/.png`:

## Memory Considerations

This script employs several techniques to manage large datasets efficiently:

- **Chunked Storage**: Data is stored and processed in manageable chunks (default 512×512 pixels)
- **Statistical Sampling**: Sankey diagrams use a 1% sample of pixels (adjustable via `sample_fraction`)
- **Visualization Downsampling**: Images are generated at reduced resolution (adjustable via `downsample_factor`)

For a full dataset covering Brazil (approximately 8,500,000 km²), the memory requirements are:
- Base memory usage: ~500MB
- Peak usage during combination: ~2-4GB (adjustable via chunk size)
- Disk space for combined dataset: ~20-30GB

## Notes and Limitations

- **Processing Time**: Full combination process may take 1-2 hours depending on input size
- **Statistical Accuracy**: Sampling introduces a small margin of error in Sankey diagrams (~1-2%)
- **Visualization Detail**: Downsampling reduces the detail visible in nationwide visualizations
- **Geographic Alignment**: Minor pixel alignment issues may occur when combining grid datasets
- **Memory Usage**: Adjust sampling and downsampling parameters for your available memory

---

Created by Leandro Meneguelli Biondo for the UBCO PhD project, 2025.



