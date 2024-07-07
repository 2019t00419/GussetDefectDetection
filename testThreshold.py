import cv2 as cv
import numpy as np
from contourID import identify_edges  # Assuming this is a custom function for edge identification

# Parameters
threshold1 = 100
threshold2 = 200
val = 190
# Load the original image
original_frame = cv.imread('test.jpg')

if original_frame is None:
    print("Error: Image not loaded. Please check the file path.")
    exit(1)

# Resize the original frame
original_frame_resized = cv.resize(original_frame, (1280, 960))
# Convert to HSV color space
hsv = cv.cvtColor(original_frame_resized, cv.COLOR_BGR2HSV)

# Extract the Value channel
v_channel = hsv[:, :, 2]
s_channel = hsv[:, :, 1]
h_channel = hsv[:, :, 0]

# Convert to grayscale
grayscale_image = cv.cvtColor(original_frame_resized, cv.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)
blurred_image1 = cv.GaussianBlur(s_channel, (5, 5), 0)

# Otsu's Binarization
_, otsu_thresholded = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
_, otsu_thresholded1 = cv.threshold(blurred_image1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

blurred_otsu = cv.GaussianBlur(otsu_thresholded, (5, 5), 0)
blurred_otsu1 = cv.GaussianBlur(otsu_thresholded1, (5, 5), 0)


# Apply Canny edge detection on the original resized image
canny = cv.Canny(blurred_otsu, threshold1, threshold2)
canny1 = cv.Canny(blurred_otsu1, threshold1, threshold2)

# Find contours and identify outer edge
contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
contours1, _ = cv.findContours(canny1, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

outer_edge, _ = identify_edges(contours)  # Assuming this function correctly identifies the outer edge
outer_edge1, _ = identify_edges(contours1)  # Assuming this function correctly identifies the outer edge


cv.drawContours(original_frame_resized, [outer_edge], -1, val, 3)
cv.drawContours(original_frame_resized, [outer_edge1], -1, val, 3)

cv.imshow("h_channel",h_channel)
cv.imshow("s_channel",s_channel)
cv.imshow("v_channel",v_channel)
cv.imshow("blurred_otsu",blurred_otsu)
cv.imshow("blurred_otsu1",blurred_otsu1)
cv.imshow("canny",canny)
cv.imshow("canny1",canny1)
cv.imshow("original_frame_resized",original_frame_resized)


cv.waitKey(0)
cv.destroyAllWindows()
