import cv2 as cv
import numpy as np
from defect_check import checkGussetPosition,checkBalanceOut,check_fabric_damage
from contourID import identify_edges,identify_outer_edge
from miscellaneous import preprocess
from sideMixupDetection import crop_image
from display_items import outputs
import time
from textureAnalysis import detect_stains
from datetime import datetime

#source= cv.VideoCapture(0)
#video_source= cv.VideoCapture("images\in\sample.mp4")


def generateOutputFrame(captured_frame,sample_longest_contour,sample_second_longest_contour,styleValue,thickness,colour,captured_time):    
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
    
    
    original_frame,blurred_otsu,assisted_defects_mask,canny,assisted_fabric_mask= preprocess(original_frame,sample_longest_contour,sample_second_longest_contour,styleValue,thickness,colour,captured_time)    
    
    frame_contours = original_frame.copy()
    #frame_contours = original_frame_resized.copy()
    processed_frame = original_frame.copy()
    #processed_frame = original_frame_resized.copy()
    
        # Find contours
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Find the longest contour
    _,longest_contour,second_longest_contour=identify_edges(contours,sample_longest_contour,sample_second_longest_contour)
    print("1")
    if longest_contour is not None:
        match_gusset_shape = cv.matchShapes(longest_contour,sample_longest_contour,1,0.0)
        if match_gusset_shape > 0.2:
            gusset_identified = False
            longest_contour = None
            print("2")
        else :
            print("3")
            gusset_identified = True
            if second_longest_contour is not None:
                print("4")
                match_fabric_shape = cv.matchShapes(second_longest_contour,sample_second_longest_contour,1,0.0)
                total_area = cv.contourArea(longest_contour)
                fabric_area = cv.contourArea(second_longest_contour)
                area_ratio = fabric_area/total_area
                if match_fabric_shape < 0.25 and area_ratio > 0.5:
                    gusset_side = "Back"
                    print("6")
                else :
                    gusset_side = "error"
                    print("7")
            else:
                print("5")
                gusset_side = "Front"
            fabric_side = crop_image(original_frame, longest_contour,100)
   
            #longest_contour = checkGussetPosition(gusset_identified,original_frame,frame_contours,original_frame_resized,longest_contour,second_longest_contour)
    else:
        print("8")
        # Processing grayscale image
        blurred_assisted_fabric_mask = cv.GaussianBlur(assisted_fabric_mask, (5, 5), 0)
        canny_check = cv.Canny(blurred_assisted_fabric_mask, 100, 200)
        contours_check, _ = cv.findContours(canny_check, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        # Find the longest contour
        longest_contour_check=identify_outer_edge(contours_check,sample_longest_contour)
        second_longest_contour_check = None

        if longest_contour_check is not None:
            match_gusset_shape = cv.matchShapes(longest_contour_check,sample_longest_contour,1,0.0)

            if match_gusset_shape > 0.2:
                gusset_identified = False
                longest_contour_check = None
                print("9")
            else :
                print("10")
                gusset_identified = True
                gusset_side = "Front"
                fabric_side = crop_image(original_frame, longest_contour_check,100)
        else:
            fabric_side = "error"



    if gusset_identified:     
        defect_contours,fabric_damage_bool=check_fabric_damage(assisted_defects_mask)

        if fabric_damage_bool :
                fabric_damage = "Damaged"
        else:
                fabric_damage = "No issue"

        if gusset_side == "Back" :
            balance_out_bool = checkBalanceOut(longest_contour,second_longest_contour,frame_contours,thickness)
            #Adding texture analysis

            #isolate adhesive
            fabric_mask_colour = np.ones_like(original_frame)
            cv.drawContours(fabric_mask_colour, [longest_contour], -1, (255,255,255), cv.FILLED)
            cv.drawContours(fabric_mask_colour, [second_longest_contour], -1, (0,0,0), cv.FILLED)

            fabric_mask = cv.cvtColor(fabric_mask_colour, cv.COLOR_BGR2GRAY)

            #cv.imshow("fabric_mask",fabric_mask)

            
            masked_image_for_texture = cv.bitwise_and(original_frame, fabric_mask_colour, mask=fabric_mask)
            #cv.imshow("masked_image_for_texture",masked_image_for_texture)
            #stain_marks = detect_stains(masked_image_for_texture)
            stain_marks = True
            if stain_marks :
                print("Stain marks are avilable")
            else:
                print("fabric status is fine")

            if balance_out_bool :
                balance_out = "Balance out"
            else:
                balance_out = "No issue"
            processed_frame=outputs(gusset_identified,gusset_side,longest_contour,second_longest_contour,frame_contours,original_frame,blurred_otsu,canny,c,fabric_damage,defect_contours)


        elif(gusset_side == "Front"):
            balance_out = "Front side of the gusset detected"
            processed_frame=outputs(gusset_identified,gusset_side,longest_contour_check,second_longest_contour_check,frame_contours,original_frame,blurred_otsu,canny,c,fabric_damage,defect_contours)


    else:
        fabric_damage = "error"
        fabric_side = "error"
        defect_contours = None
        processed_frame=outputs(gusset_identified,gusset_side,longest_contour,second_longest_contour,frame_contours,original_frame,blurred_otsu,canny,c,fabric_damage,defect_contours)

    
            
        # End of time calculation
    end_time = time.time()  # End time
    elapsed_time = (end_time - start_time)*1000  # Calculate elapsed time
    print(f"Time taken to generate output frame: {elapsed_time:.4f} ms\n\n") 

        # Format: YYYYMMDD_HHMMSS
    
    cv.imwrite(f"images/captured/processed/processed ({captured_time}).jpg", processed_frame)
    cv.imwrite(f"images/captured/original/original ({captured_time}).jpg", original_frame)

    return processed_frame,balance_out,fabric_side,gusset_side,fabric_damage,blurred_otsu

