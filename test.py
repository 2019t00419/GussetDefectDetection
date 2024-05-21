import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
image_path = 'gusset_1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Otsu's Binarization
_, otsu_thresholded = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Otsu's Binarization")
plt.imshow(otsu_thresholded, cmap='gray')

plt.show()
