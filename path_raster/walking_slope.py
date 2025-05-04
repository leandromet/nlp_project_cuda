# Load the slope image
slope_image_path = "/mnt/data/example_kelowna_slope_enhance.png"
slope_image = cv2.imread(slope_image_path)

# Convert to grayscale (slope intensity assumed to be in visual brightness)
slope_gray = cv2.cvtColor(slope_image, cv2.COLOR_BGR2GRAY)

# Threshold to extract flat/low-slope areas (likely walkable)
# The threshold may need tuning: lower values capture flatter regions
_, flat_zones = cv2.threshold(slope_gray, 50, 255, cv2.THRESH_BINARY_INV)

# Morphological operations to clean up and connect paths
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
paths_morph = cv2.morphologyEx(flat_zones, cv2.MORPH_CLOSE, kernel, iterations=2)
paths_morph = cv2.morphologyEx(paths_morph, cv2.MORPH_OPEN, kernel, iterations=1)

# Overlay extracted probable paths on the slope image
slope_overlay = slope_image.copy()
slope_overlay[paths_morph == 255] = [0, 255, 0]  # Green for walkable paths

# Convert for display
slope_overlay_rgb = cv2.cvtColor(slope_overlay, cv2.COLOR_BGR2RGB)

# Plot
plt.figure(figsize=(16, 10))
plt.imshow(slope_overlay_rgb)
plt.title("Probable Walking Paths from Slope (in Green)")
plt.axis("off")
plt.show()
