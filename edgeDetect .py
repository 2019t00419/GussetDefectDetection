import cv2 as cv
import numpy as np
import os
from fillCoordinates import fill_coordinates

#camera= cv.VideoCapture(0)

# Check if the file exists
file_path = 'gusset.jpg'
if not os.path.exists(file_path):
    print("Error: File '{}' not found.".format(file_path))
    exit()

# Read the image
original_frame = cv.imread(file_path)
threshold1=200
threshold2=400
while True:
    
    #_,original_frame=camera.read()
    #cv.imshow('Original Image', original_frame)

    # Apply Laplacian edge detection
    laplace = cv.Laplacian(original_frame, cv.CV_64F)
    laplace = np.uint8(np.absolute(laplace))
    #cv.imshow('Laplacian Edge', laplace)

    # Apply Canny edge detection
    canny = cv.Canny(original_frame, threshold1, threshold2)
    #cv.imshow('Canny Edge', canny)

    # Find contours
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Find the longest contour
    max_length = 0
    second_max_length = 0
    longest_contour = None
    second_longest_contour = None

    for contour in contours:
        length = cv.arcLength(contour, closed=True)
        if length > max_length:
            max_length = length  
            longest_contour = contour
        elif length > second_max_length:
            second_max_length = length
            second_longest_contour = contour
        
    # Highlight the longest edge
    frame_contours = original_frame.copy()

    if longest_contour is not None:
        cv.drawContours(frame_contours, [longest_contour], -1, (0, 255, 0), 3)
        longest_contour=fill_coordinates(longest_contour)
        for coordinates in longest_contour:
            x, y = coordinates[0]  # Extract x and y coordinates from the point
            cv.circle(frame_contours, (x, y), 1, (0, 0, 0), -1)

    if second_longest_contour is not None:    
        cv.drawContours(frame_contours, [second_longest_contour], -1, (0, 0, 255), 3)
        second_longest_contour=fill_coordinates(second_longest_contour)
        for coordinates in second_longest_contour:
            x, y = coordinates[0]  # Extract x and y coordinates from the point
            cv.circle(frame_contours, (x, y), 1, (0, 0, 0), -1)


        cv.imshow('Longest Edge', frame_contours)        
    else:
        cv.imshow('Longest Edge', original_frame)
        print("Invalid contours")
    cv.imwrite("Output.jpg",frame_contours)
    # Wait for 'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break

# Release resources
cv.destroyAllWindows()
