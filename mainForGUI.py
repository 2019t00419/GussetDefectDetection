import cv2 as cv
import numpy as np
from balanceOut import checkBalanceOut
from balanceOut import outputs
from contourID import identify_edges
from miscellaneous import openFile
from miscellaneous import camera
from miscellaneous import preprocess_cpu
from miscellaneous import preprocess_gpu
from fabricDefects import fabric_color
from new import new_feature
import time
from miscellaneous import log

# Check if the file exists
sample_path = "images\sample\sample (1).jpg"
source= cv.VideoCapture(0)
#video_source= cv.VideoCapture("images\in\sample.mp4")

cv.cuda.setDevice(0)

gaussian_filter = cv.cuda.createGaussianFilter(0,0, (5, 5), 0) #cv.cuda.createGaussianFilter(input type,output type, kernal, 0)

canny_edge_detector = cv.cuda.createCannyEdgeDetector(low_thresh=50, high_thresh=150)

gpu_original_frame = cv.cuda_GpuMat()


def main(captured_frame,gpu): 
    
    start_time = time.time()  # Start timex   
    c=0
    #chose read image mode
    original_frame = captured_frame
    #original_frame = camera(video_source)
    #original_frame = cv.rotate(original_frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    if gpu:
        blurred_otsu,canny = preprocess_gpu(original_frame,gpu_original_frame,gaussian_filter,canny_edge_detector,c)  
    else:
        blurred_otsu,canny = preprocess_cpu(original_frame,c)  

    frame_contours = original_frame.copy()
    
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
               
    
    longest_contour = checkBalanceOut(original_frame,frame_contours,longest_contour,second_longest_contour)

    out=outputs(longest_contour,second_longest_contour,frame_contours,original_frame,blurred_otsu,canny,c)
        
    end_time = time.time()  # End time
    # End of time calculation
    elapsed_time = (end_time - start_time)*1000  # Calculate elapsed time
    if gpu:
        log(f"GPU : Time taken to complete the function: {elapsed_time:.4f} ms\n")
        print(f"GPU : Time taken to complete the function: {elapsed_time:.4f} ms\n\n") 
    else:
        log(f"CPU : Time taken to complete the function: {elapsed_time:.4f} ms\n")
        print(f"CPU : Time taken to complete the function: {elapsed_time:.4f} ms\n\n")

    return out

