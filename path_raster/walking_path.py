import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Load image
image_path = "/mnt/data/example_kelowna.png"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute gradient using Sobel filters (to get slope)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

# Threshold low gradient areas (flat areas)
_, flat_areas = cv2.threshold(gradient_magnitude, 20, 255, cv2.THRESH_BINARY_INV)

# Morphological operations to enhance linear structures
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morphed = cv2.morphologyEx(flat_areas, cv2.MORPH_CLOSE, kernel, iterations=2)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)

# Overlay detected flat zones on the original image
overlay = image.copy()
overlay[morphed == 255] = [0, 255, 0]  # Mark probable walking paths in green

# Convert BGR to RGB for displaying with matplotlib
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# Display results
plt.figure(figsize=(16, 10))
plt.imshow(overlay_rgb)
plt.title("Probable Walking Paths (in Green)")
plt.axis("off")
plt.show()
