import numpy as np
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

# Load the image
image = io.imread('test.jpg')

# Convert to grayscale
gray_image = color.rgb2gray(image)

# Quantize the grayscale image to reduce the number of gray levels
gray_image = (gray_image * 255).astype(np.uint8)

# Compute the GLCM
distances = [1]  # distance between pixel pairs
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles in radians
glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

# Extract GLCM properties
properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# Calculate the properties
for prop in properties:
    print(f'{prop}: {graycoprops(glcm, prop)}')

# Visualize the GLCM
plt.figure(figsize=(8, 8))
for i, angle in enumerate(angles):
    plt.subplot(2, 2, i+1)
    plt.imshow(glcm[:, :, 0, i], cmap='gray')
    plt.title(f'GLCM for angle {angle} radians')
plt.tight_layout()
plt.show()
