import cv2 as cv
import numpy as np
import os
from fillCoordinates import fill_coordinates
from fillCoordinates import measure_distance
from fillCoordinates import display
import time

#camera= cv.VideoCapture(0)

# Check if the file exists
file_path = 'gusset (5).jpg'
if not os.path.exists(file_path):
    print("Error: File '{}' not found.".format(file_path))
    exit()

# Read the image
original_frame = cv.imread(file_path)
original_frame = cv.resize(original_frame, (960, 1280))
original_frame_resized = cv.resize(original_frame, (960, 1280))
grayscale_image = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)

# Otsu's Binarization
_, otsu_thresholded = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

otsu_resized = cv.resize(otsu_thresholded, (960, 1280))       
cv.imshow('otsu_thresholded', otsu_resized)

threshold1=200
threshold2=300
while True:
    
    #_,original_frame=camera.read()
    #cv.imshow('Original Image', original_frame)


    # Apply Canny edge detection
    canny = cv.Canny(otsu_thresholded, threshold1, threshold2)
    canny_resized = cv.resize(canny, (960, 1280))
    cv.imshow('Canny Edge', canny_resized)

    # Find contours
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Find the longest contour
    max_length = 0
    second_max_length = 0
    longest_contour = None
    second_longest_contour = None
    relative_length0 = 0
    relative_length1 = 0

    #start_time = time.time()

    for contour in contours:
        length = cv.arcLength(contour, closed=True)
        print(str(length))
        if length > max_length:
            if max_length != 0:
                relative_length1=(length-max_length)/max_length
                if 0.05 < relative_length1 < 0.5:
                    second_max_length = max_length
                    second_longest_contour = longest_contour
                    #print("longest : "+str(max_length)+" second longest : "+str(second_max_length)+ " relative : "+str(relative_length1))
            else:
                second_max_length = max_length
                second_longest_contour = longest_contour
            max_length = length  
            longest_contour = contour

        elif length > second_max_length:
            if second_max_length != 0:
                relative_length1=(max_length-length)/length
                if 0.05 < relative_length1 < 0.5:
                    second_max_length = length
                    second_longest_contour = contour
                    #print("longest : "+str(max_length)+" second longest : "+str(second_max_length)+ " relative : "+str(relative_length1))
    print("longest : "+str(max_length)+" second longest : "+str(second_max_length)+ " relative : "+str(relative_length1))                


    #end_time = time.time()

    #print("Execution time = "+str(1000*(end_time-start_time))+"ms")
        
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
        
        frame_contours_resized = cv.resize(frame_contours, (960, 1280))
        cv.imshow('Edges', frame_contours_resized) 
        
        cv.imwrite("Output.jpg",frame_contours)
        cv.imwrite("otsu_thresholded.jpg",otsu_thresholded)
        cv.imwrite("canny.jpg",canny)
    else:
        cv.imshow('Edges', original_frame)
        print("Invalid contours")
    # Wait for 'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break

# Release resources
cv.destroyAllWindows()