import cv2 as cv
import numpy as np
import os
from fillCoordinates import fill_coordinates
from balanceOut import measure_distance_KDTree
from contourID import identify_edges
import time

#camera= cv.VideoCapture(0)

# Check if the file exists

threshold1=100
threshold2=200
c=1        
defect_count=0
non_defect_count=0

while True:    
    start_time = time.time()  # Start time
    
    file_path = "in\gusset ("+str(c)+").jpg"
    if not os.path.exists(file_path):
        print("Error: File '{}' not found.".format(file_path))
        exit()

    # Read the image
    original_frame = cv.imread(file_path)


    frame_height, frame_width, channels = original_frame.shape
    resolution_factor = int(((frame_height ** 2) + (frame_width ** 2)) ** 0.5)
    #print("Resolution of the image is : "+str((frame_height*frame_width)/1000000)+"MP")
    #print("Resolution factor is : "+str(resolution_factor))


    original_frame = cv.resize(original_frame, (960, 1280))
    original_frame_resized = cv.resize(original_frame, (960, 1280))
    grayscale_image = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)

    # Otsu's Binarization
    _, otsu_thresholded = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(otsu_thresholded, (5, 5), 0)
    otsu_resized = cv.resize(blurred_otsu, (960, 1280))    
     
    cv.imshow('otsu_thresholded', otsu_resized)
        #_,original_frame=camera.read()
        #cv.imshow('Original Image', original_frame)


    # Apply Canny edge detection
    canny = cv.Canny(blurred_otsu, threshold1, threshold2)
    canny_resized = cv.resize(canny, (960, 1280))

    cv.imshow('Canny Edge', canny_resized)


    
    # Find contours
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Find the longest contour
    longest_contour,second_longest_contour=identify_edges(contours)

        
    # Highlight the longest edge
    frame_contours = original_frame.copy()
    
    if longest_contour is not None:
        #draw contours on to the frame
        cv.drawContours(frame_contours, [longest_contour], -1, (0, 255, 0), 3)
        x_bound,y_bound,w_bound,h_bound = cv.boundingRect(longest_contour)
        cv.rectangle(frame_contours,(x_bound,y_bound),(x_bound+w_bound,y_bound+h_bound),(0,255,0),2)
        #complete the incomplete coordinates
        #longest_contour=fill_coordinates(longest_contour)
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
        #second_longest_contour=fill_coordinates(second_longest_contour)
        #plot the coordinates
        for coordinates in second_longest_contour:
            x, y = coordinates[0]  # Extract x and y coordinates from the point
            cv.circle(frame_contours, (x, y), 1, (0, 0, 0), -1)

    if second_longest_contour is not None and longest_contour is not None:
        
        if measure_distance_KDTree(longest_contour,second_longest_contour,frame_contours):
            defect_count=defect_count+1
        else :
            non_defect_count=non_defect_count+1
        
        total_area = cv.contourArea(longest_contour)
        fabric_area = cv.contourArea(second_longest_contour)
        adhesive_area = total_area - fabric_area
        cv.putText(frame_contours, "Total area : "+str(total_area), (400, 625), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
        cv.putText(frame_contours, "Fabric_area : "+str(fabric_area), (400, 650), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
        cv.putText(frame_contours, "Adhesive area : "+str(adhesive_area), (400, 675), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
        #display(frame_contours,longest_contour)
        
        frame_contours_resized = cv.resize(frame_contours, (960, 1280))
        cv.imshow('Edges', frame_contours_resized) 
        
        cv.imwrite("out\output\Output ("+str(c)+").jpg",frame_contours)
        cv.imwrite("out\otsu\otsu ("+str(c)+").jpg",otsu_thresholded)
        cv.imwrite("out\otsu\otsu_b ("+str(c)+").jpg",blurred_otsu)
        cv.imwrite("out\canny\canny ("+str(c)+").jpg",canny)
        print("Defect count :"+str(defect_count)+"\t Non defect count :"+str(non_defect_count))
    else:
        cv.imshow('Edges', original_frame)
        print("Invalid contours")
    # Wait  'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break

    c=c+1
    
    # End of time calculation
    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Time taken to complete the function: {elapsed_time:.4f} seconds") 
    
# Release resources
cv.destroyAllWindows()