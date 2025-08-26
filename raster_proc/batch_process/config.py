# -*- coding: utf-8 -*-
"""
Configuration module for raster processing and visualization.

Contains all constants, color mappings, and configuration parameters
used throughout the land cover analysis pipeline.
"""

import logging
import psutil

# Processing Configuration
TILE_SIZE = 256
MAX_WORKERS = 4  # Reduced to prevent memory issues and system lockup
MEMORY_BUFFER_GB = 5
MAX_READ_RETRIES = 3
RETRY_DELAY = 0.05  # seconds
VRT_BLOCK_SIZE = 512  # Matches the BlockXSize/BlockYSize in VRT
PROCESSING_TILE_SIZE = 2048  # Should be a multiple of VRT_BLOCK_SIZE
FILL_VALUE = 0  # Matches NoDataValue in VRT

# Color mapping for land cover classes
COLOR_MAP = {
    0: "#ffffff",
    1: "#1f8d49", 3: "#1f8d49", 4: "#7dc975", 5: "#04381d", 6: "#007785",
    9: "#7a5900", 10: "#d6bc74", 11: "#519799", 12: "#d6bc74", 13: "#ffffff", 14: "#ffefc3",
    15: "#edde8e", 18: "#e974ed", 19: "#c27ba0", 20: "#db7093",   
    21: "#ffefc3", 22: "#d4271e", 23: "#ffa07a", 24: "#d4271e", 25: "#db4d4f", 26: "#2532e4", 29: "#ffaa5f",
    30: "#9c0027", 31: "#091077", 32: "#fc8114", 33: "#259fe4", 35: "#9065d0", 36: "#d082de",
    39: "#f5b3c8", 40: "#c71585", 41: "#f54ca9", 46: "#d68fe2", 47: "#9932cc",
    48: "#e6ccff", 49: "#02d659", 50: "#ad5100", 62: "#ff69b4", 27: "#ffffff"
}

# Labels for land cover classes
LABELS = {
    0: "No data", 
    1: "Forest", 3: "Forest Formation", 4: "Savanna Formation", 5: "Mangrove", 6: "Floodable Forest",
    9: "Forest Plantation", 11: "Wetland", 10: "Herbaceous", 12: "Grassland", 13: "other", 14: "Farming",
    15: "Pasture", 18: "Agri", 19: "Temporary Crop", 20: "Sugar Cane",
    21: "Mosaic of Uses", 22: "Non vegetated", 23: "Beach and Sand", 24: "Urban Area",
    25: "Other non Vegetated Areas", 26: "Water", 29: "Rocky Outcrop", 30: "Mining", 31: "Aquaculture",
    32: "Hypersaline Tidal Flat", 33: "River Lake and Ocean", 35: "Palm Oil", 36: "Perennial Crop", 39: "Soybean",
    40: "Rice", 41: "Other Temporary Crops", 46: "Coffee", 47: "Citrus", 48: "Other Perennial Crops",
    49: "Wooded Sandbank Vegetation", 50: "Herbaceous Sandbank Vegetation", 62: "Cotton", 27: "Not Observed"
}

# Analysis periods
DECADES = [
    (1985, 1994),  # First decade
    (1995, 2004),  # Second decade
    (2005, 2014),  # Third decade
    (2015, 2024),  # Fourth period
    (1985, 2024)   # Full period
]

def get_available_memory():
    """Calculate available memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
import logging

