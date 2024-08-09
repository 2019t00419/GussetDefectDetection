import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

# Load the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute GLCM
glcm = greycomatrix(image, [1], [0], 256, symmetric=True, normed=True)

# Extract texture features
contrast = greycoprops(glcm, 'contrast')
dissimilarity = greycoprops(glcm, 'dissimilarity')
homogeneity = greycoprops(glcm, 'homogeneity')
energy = greycoprops(glcm, 'energy')
correlation = greycoprops(glcm, 'correlation')

# Use features for segmentation (e.g., thresholding, clustering, etc.)
# Example: Simple thresholding on the 'contrast' feature
_, segmented_image = cv2.threshold(contrast, 0.5, 255, cv2.THRESH_BINARY)

# Save or display the segmented image
cv2.imwrite('segmented_image.jpg', segmented_image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
