import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk

def apply_entropy(gray_image, neighborhood_radius=3):
    # Convert to grayscale
    
    # Apply entropy filter
    entropy_image = entropy(gray_image, disk(neighborhood_radius))
    
    # Normalize entropy for visualization
    entropy_normalized = cv2.normalize(entropy_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return entropy_normalized

image = cv2.imread('images/in/masked/Test_Image_ (27).jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
entropy_output = apply_entropy(gray_image)
entropy_outputi = 255 - entropy_output
entropy_output1 = apply_entropy(entropy_outputi)
entropy_output = cv2.resize(entropy_output, (960, 1280))
entropy_output1 = cv2.resize(entropy_output1, (960, 1280))
cv2.imshow('Entropy Output', entropy_output)
cv2.imshow('Entropy Output1', entropy_output1)

cv2.waitKey(0)
cv2.destroyAllWindows()
