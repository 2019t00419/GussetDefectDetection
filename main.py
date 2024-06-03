import cv2 as cv
import numpy as np
from balanceOut import checkBalanceOut
from balanceOut import outputs
from contourID import identify_edges
from miscellaneous import openFile
from miscellaneous import camera
from miscellaneous import preprocess
from fabricDefects import fabric_color
from new import new_feature
import time

# Check if the file exists
c=1        
sample_path = "images\sample\sample (1).jpg"
source= cv.VideoCapture(0)
video_source= cv.VideoCapture("images\in\sample.mp4")

while True:    
    
    start_time = time.time()  # Start timex
    #chose read image mode
    original_frame = cv.imread(openFile(c))
    #original_frame = camera(source)
    #original_frame = cv.rotate(original_frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    original_frame,original_frame_resized,blurred_otsu,canny,blurred_image,grayscale_image = preprocess(original_frame,c)    
    frame_contours = original_frame_resized.copy()
    
    # Find contours
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Find the longest contour
    longest_contour,second_longest_contour=identify_edges(contours)
    ret = cv.matchShapes(longest_contour,second_longest_contour,1,0.0)
    #print(ret)
    if(longest_contour is not None):    
        new_feature(original_frame,longest_contour,c)

    if(second_longest_contour is not None):
        total_area = cv.contourArea(longest_contour)
        fabric_area = cv.contourArea(second_longest_contour)
        area_ratio = fabric_area/total_area
        #fabric_color(original_frame,second_longest_contour,c)
    else:
        area_ratio=0

    if(ret>0.5) or (area_ratio < 0.5) :
        longest_contour=None
        #print("dissimilar")
    else:
        #print("Similar") 
        area_ratio=area_ratio
               
    
    longest_contour = checkBalanceOut(original_frame,frame_contours,original_frame_resized,longest_contour,second_longest_contour)

    outputs(longest_contour,second_longest_contour,frame_contours,original_frame,original_frame_resized,blurred_otsu,canny,c)
        
    # Highlight the longest edge
    
    
    # Wait  'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break

    c=c+1
    # End of time calculation
    end_time = time.time()  # End time
    elapsed_time = (end_time - start_time)*1000  # Calculate elapsed time
    print(f"Time taken to complete the function: {elapsed_time:.4f} ms\n\n") 
    
# Release resources
cv.destroyAllWindows()

