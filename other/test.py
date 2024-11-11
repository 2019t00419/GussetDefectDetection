import cv2
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np

# Load image
image = io.imread('Images\\captured\\Original\\test.jpg')
gray_image = color.rgb2gray(image)
gray_image = (gray_image * 255).astype(np.uint8)

# Apply Gaussian blur to reduce high-frequency noise, but preserve edges
blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)  # Increased kernel size to better handle noise

# Apply Canny edge detection with adjusted thresholds
lower_threshold = 50  # Try a lower threshold for softer edges
upper_threshold = 100  # Keep the upper threshold high enough to avoid noise
edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)

# Refined highlighting of defects: only focus on strong edges by increasing contrast
highlighted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
highlighted_image[edges == 255] = [0, 0, 255]  # Highlight detected edges in red

# Visualize the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.title("Edge Map (Canny)")
plt.imshow(edges, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Highlighted Defects")
plt.imshow(highlighted_image)
plt.tight_layout()
plt.show()
