import cv2 as cv
import numpy as np
from balanceOut import checkGussetPosition,checkBalanceOut
from contourID import identify_edges
from miscellaneous import preprocess
from SMDYOLO import crop_image
from display_items import outputs
import time

source= cv.VideoCapture(0)
#video_source= cv.VideoCapture("images\in\sample.mp4")


def generateOutputFrame(captured_frame,style,sample_longest_contour,sample_second_longest_contour):    
    c=0
    gusset_identified = False
    gusset_side = "Not identified"
    processed_frame = None
    balance_out = "Error"
    fabric_side = "error"

    start_time = time.time()  # Start timex
    #chose read image mode
    original_frame = captured_frame
    #original_frame = camera(video_source)
    #original_frame = cv.rotate(original_frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    original_frame,original_frame_resized,blurred_otsu,canny,blurred_image,grayscale_image = preprocess(original_frame,style,sample_longest_contour,sample_second_longest_contour)    
    frame_contours = original_frame_resized.copy()
    
    processed_frame = original_frame_resized.copy()
    

    
    # Find contours
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Find the longest contour
    _,longest_contour,second_longest_contour=identify_edges(contours,sample_longest_contour,sample_second_longest_contour)
    
    if longest_contour is not None:
        match_gusset_shape = cv.matchShapes(longest_contour,sample_longest_contour,1,0.0)
        if match_gusset_shape > 0.2:
            gusset_identified = False
            longest_contour = None
        else :
            gusset_identified = True
            if second_longest_contour is not None:
                match_fabric_shape = cv.matchShapes(second_longest_contour,sample_second_longest_contour,1,0.0)
                total_area = cv.contourArea(longest_contour)
                fabric_area = cv.contourArea(second_longest_contour)
                area_ratio = fabric_area/total_area
                if match_fabric_shape < 0.25 and area_ratio > 0.5:
                    gusset_side = "Back"
                else :
                    gusset_side = "Front"
            else:
                gusset_side = "Front"
   
            #longest_contour = checkGussetPosition(gusset_identified,original_frame,frame_contours,original_frame_resized,longest_contour,second_longest_contour)
        
        if gusset_identified:
            if gusset_side == "Back" :
                balance_out_bool = checkBalanceOut(longest_contour,second_longest_contour,frame_contours)
                if balance_out_bool :
                    balance_out = "Balance out"
                else:
                    balance_out = "No issue"

            else :
                balance_out = "Front side of the gusset detected"

            fabric_side = crop_image(original_frame, longest_contour, 0)
        else:
            fabric_side = "error"

        processed_frame=outputs(gusset_identified,gusset_side,longest_contour,second_longest_contour,frame_contours,original_frame,original_frame_resized,blurred_otsu,canny,c)
                
            # End of time calculation
        end_time = time.time()  # End time
        elapsed_time = (end_time - start_time)*1000  # Calculate elapsed time
        print(f"Time taken to complete the function: {elapsed_time:.4f} ms\n\n") 
    return processed_frame,balance_out,fabric_side,gusset_side




