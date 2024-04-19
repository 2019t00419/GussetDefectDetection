import cv2 as cv
import numpy as np
import os

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
    cv.imshow('Original Image', original_frame)

    # Apply Laplacian edge detection
    laplace = cv.Laplacian(original_frame, cv.CV_64F)
    laplace = np.uint8(np.absolute(laplace))
    cv.imshow('Laplacian Edge', laplace)

    # Apply Canny edge detection
    canny = cv.Canny(original_frame, threshold1, threshold2)
    cv.imshow('Canny Edge', canny)

    # Find contours
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    print(len(contours))

    # Find the longest contour
    max_length = 0
    second_max_length = 0
    longest_contour = None
    second_longest_contour = None

    for contour in contours:
        length = cv.arcLength(contour, closed=True)
        print(length)
        if length > max_length:
            max_length = length  
            longest_contour = contour
        elif length > second_max_length:
            second_longest_contour = contour

    if second_longest_contour is not None:     
        print("second longest edge is available")
    else:
        print("second longest edge is not available")
        
    # Highlight the longest edge
    frame_with_longest_contour = original_frame.copy()
    frame_with_second_longest_contour = original_frame.copy()

    if longest_contour is not None:
        cv.drawContours(frame_with_longest_contour, [longest_contour], -1, (0, 255, 0), 2)
        cv.imshow('Longest Edge', frame_with_longest_contour)        
    else:
        cv.imshow('Longest Edge', original_frame)
    if second_longest_contour is not None:
        cv.drawContours(frame_with_second_longest_contour, [second_longest_contour], -1, (0, 0, 255), 2)
        cv.imshow('second_Longest Edge', frame_with_second_longest_contour)        
    else:
        cv.imshow('second_Longest Edge', original_frame)

    # Wait for 'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break

# Release resources
cv.destroyAllWindows()
