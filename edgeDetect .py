import cv2 as cv
import numpy as np
import os
from fillCoordinates import fill_coordinates
from fillCoordinates import measure_distance
from fillCoordinates import display

#camera= cv.VideoCapture(0)

# Check if the file exists
file_path = 'gusset.jpg'
if not os.path.exists(file_path):
    print("Error: File '{}' not found.".format(file_path))
    exit()

# Read the image
original_frame = cv.imread(file_path)
threshold1=200
threshold2=300
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
        #draw contours on to the frame
        cv.drawContours(frame_contours, [longest_contour], -1, (0, 255, 0), 3)
        #complete the incomplete coordinates
        longest_contour=fill_coordinates(longest_contour)
        #plot the coordinates
        for coordinates in longest_contour:
            x, y = coordinates[0]  # Extract x and y coordinates from the point
            cv.circle(frame_contours, (x, y), 1, (0, 0, 0), -1)
    else:
        cv.imshow('Edges', original_frame)
        print("Invalid contours")

    if second_longest_contour is not None: 
        #draw contours on to the frame   
        cv.drawContours(frame_contours, [second_longest_contour], -1, (0, 0, 255), 3)
        #complete the incomplete coordinates
        second_longest_contour=fill_coordinates(second_longest_contour)
        #plot the coordinates
        for coordinates in second_longest_contour:
            x, y = coordinates[0]  # Extract x and y coordinates from the point
            cv.circle(frame_contours, (x, y), 1, (0, 0, 0), -1)
        
         

    if second_longest_contour is not None and longest_contour is not None:
        
        measure_distance(longest_contour,second_longest_contour,frame_contours)
        #display(frame_contours,longest_contour)
        cv.imshow('Edges', frame_contours) 
        cv.imwrite("Output.jpg",frame_contours)
    else:
        cv.imshow('Edges', original_frame)
        print("Invalid contours")
    # Wait for 'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break

# Release resources
cv.destroyAllWindows()
