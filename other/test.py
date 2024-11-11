import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_oriented_sobel(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the Sobel gradients in the x and y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the gradient magnitude and angle
    magnitude = cv2.magnitude(sobelx, sobely)
    angle = cv2.phase(sobelx, sobely, angleInDegrees=True)
    
    # Normalize magnitude to the range [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Map angles to hue values for coloring
    hsv = np.zeros((*angle.shape, 3), dtype=np.uint8)
    hsv[..., 0] = angle / 2           # OpenCV hue range is [0, 180] for 0-360 degrees
    hsv[..., 1] = 255                 # Full saturation
    hsv[..., 2] = cv2.convertScaleAbs(magnitude)  # Use magnitude as the value
    
    # Convert HSV to RGB for display
    colored_edges = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return colored_edges

# Load an image
image = cv2.imread('Images\\captured\\original\\original (20241107_153813).jpg')

# Apply the color-oriented Sobel filter
colored_edges = color_oriented_sobel(image)

# Display the result
plt.imshow(colored_edges)
plt.axis('off')
plt.title("Edge Orientation using Sobel and Colors")
plt.show()
