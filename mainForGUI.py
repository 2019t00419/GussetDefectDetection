import cv2 as cv
import numpy as np
from defect_check import checkGussetPosition,checkBalanceOut,check_fabric_damage,checkPanelCutDamage
from contourID import identify_edges,identify_outer_edge
from miscellaneous import preprocess
from sideMixupDetection import crop_image
from display_items import outputs
import time
#from textureAnalysis import detect_stains
from datetime import datetime



#source= cv.VideoCapture(0)
#video_source= cv.VideoCapture("images\in\sample.mp4")


def generateOutputFrame(captured_frame,sample_longest_contour,sample_second_longest_contour,styleValue,adhesiveWidth,colour,captured_time):    
    defects =[]
    gusset_identified = False
    processed_frame = None
    gusset_side = "Not identified"
    balance_out = "Error"
    fabric_side = "error"
    panel_cut_damage = "error"
    longest_contour = None
    second_longest_contour = None
    longest_contour_check = None
    printY = 300

    start_time = time.time()  # Start timex
    #chose read image mode
    original_frame = captured_frame
    #original_frame = camera(video_source)
    #original_frame = cv.rotate(original_frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    
    original_frame,blurred_otsu,assisted_defects_mask,canny,assisted_fabric_mask= preprocess(original_frame,sample_longest_contour,sample_second_longest_contour,styleValue,adhesiveWidth,colour,captured_time)    
    
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

        longest_contour_area = cv.contourArea(longest_contour)

        match_longest_contour_shape = cv.matchShapes(longest_contour,sample_longest_contour,1,0.0)

        if longest_contour_area > 200: # longest_contour_area is OK
            gusset_identified = True
            fabric_side = crop_image(original_frame, longest_contour,100)
            if match_longest_contour_shape < 0.2: # match_longest_contour_shape  is OK
                #longest contour is OK
                if second_longest_contour is not None:
                    second_longest_contour_area = cv.contourArea(second_longest_contour)
                    match_second_longest_contour_shape = cv.matchShapes(second_longest_contour,sample_second_longest_contour,1,0.0)
                    if second_longest_contour_area > 200: #second_longest_contour_area is OK
                        if match_second_longest_contour_shape < 0.2: # match_second_longest_contour_shape  is OK
                            #Adhesive is OK
                            print("Adhesive is Okay")
                            gusset_side = "Back"
                        else:
                            #Ahsesive is defective
                            print("Adhesive is Defective")
                            gusset_side = "defective"
                            defects.append("Adhesive is Defective")
                    else:
                        #Ahsesive is defective
                        print("Adhesive is Defective")
                        gusset_side = "defective"
                        defects.append("Adhesive is Defective")
                else:
                    print("Adhesive shape is defective")
                    gusset_side = "defective"
                    defects.append("Adhesive shape is defective")
            else:
                print("Panel cut damage")
                gusset_side = "defective"
                defects.append("Panel cut damage")
                
        else :
            print("Noise detected. Check for Front")
            # Processing grayscale image
            blurred_assisted_fabric_mask = cv.GaussianBlur(assisted_fabric_mask, (5, 5), 0)
            canny_check = cv.Canny(blurred_assisted_fabric_mask, 100, 200)
            contours_check, _ = cv.findContours(canny_check, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            # Find the longest contour
            longest_contour_check=identify_outer_edge(contours_check,sample_longest_contour)
            second_longest_contour_check = None

            if longest_contour_check is not None:
                longest_contour_check_area = cv.contourArea(longest_contour_check)
                match_longest_contour_shape = cv.matchShapes(longest_contour_check,sample_longest_contour,1,0.0)

                if longest_contour_check_area > 200:
                    gusset_identified = True
                    fabric_side = crop_image(original_frame, longest_contour_check,100)
                    if match_longest_contour_shape < 0.2:
                        print("Front side identifed. No shape defects")
                        gusset_side = "Front"
                    else:
                        print("Front side identifed. Defecive shape")
                        gusset_side = "defective"
                        defects.append("Defecive shape")
                else:
                    print("No gusset identified. Consider as Noise")
                    gusset_identified = False
                         
            else:
                print("No fabric contours identified.")
                gusset_identified = False
                longest_contour_check = None

            
    else:
        print("No contours detected in Adhesive mask. Check for Front side")
        # Processing grayscale image
        blurred_assisted_fabric_mask = cv.GaussianBlur(assisted_fabric_mask, (5, 5), 0)
        canny_check = cv.Canny(blurred_assisted_fabric_mask, 100, 200)
        contours_check, _ = cv.findContours(canny_check, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        # Find the longest contour
        longest_contour_check=identify_outer_edge(contours_check,sample_longest_contour)

        if longest_contour_check is not None:
            longest_contour_check_area = cv.contourArea(longest_contour_check)
            match_longest_contour_shape = cv.matchShapes(longest_contour_check,sample_longest_contour,1,0.0)

            if longest_contour_check_area > 200:
                if match_longest_contour_shape < 0.2:
                    print("Front side identifed. No shape defects")
                    gusset_identified = True
                    gusset_side = "Front"
                    fabric_side = crop_image(original_frame, longest_contour_check,100)
                else:
                    print("Front side identifed. Defecive shape")
                    gusset_identified = True
                    gusset_side = "defective"
                    defects.append("Defecive shape")
                    fabric_side = crop_image(original_frame, longest_contour_check,100)
            else:
                print("No gusset identified. Consider as Noise")
                gusset_identified = False
                        
        else:
            print("No fabric contours identified.")
            gusset_identified = False
            longest_contour_check = None



    if gusset_identified:     
        fabric_damage_bool,frame_contours=check_fabric_damage(assisted_defects_mask,frame_contours)

        if fabric_damage_bool :
                fabric_damage = "Damaged"
                defects.append("fabric damage")
        else:
                fabric_damage = "No issue"

        if gusset_side == "Back" :
            balance_out_bool,printY,frame_contours = checkBalanceOut(longest_contour,second_longest_contour,frame_contours,adhesiveWidth,printY)
            panel_cut_damage_bool,frame_contours= checkPanelCutDamage(longest_contour,assisted_fabric_mask,frame_contours)
            #Adding texture analysis 
            if panel_cut_damage_bool :
                panel_cut_damage = "Panal cut damage"
                defects.append("Panal cut damage")
            else:
                panel_cut_damage = "No issue"
            print("panel_cut_damage : ",panel_cut_damage)

            if balance_out_bool :
                balance_out = "Balance out"
                defects.append("Balance out")
            else:
                balance_out = "No issue"

        elif(gusset_side == "Front"):
            balance_out = "Front side of the gusset detected"
    else:
        fabric_damage = "error"
        fabric_side = "error"
        defect_contours = None


    processed_frame=outputs(gusset_identified,gusset_side,longest_contour,second_longest_contour,longest_contour_check,frame_contours,original_frame,blurred_otsu,canny,defects,printY)    

    
    
            
        # End of time calculation
    end_time = time.time()  # End time
    elapsed_time = (end_time - start_time)*1000  # Calculate elapsed time
    print(f"Time taken to generate output frame: {elapsed_time:.4f} ms\n\n") 
    print("\n\n")
    print("balance_out : ",balance_out)
    print("fabric_damage : ",fabric_damage)
    print("gusset_side : ",gusset_side)
    print("fabric_side : ",fabric_side)
    print("\n\n")

        # Format: YYYYMMDD_HHMMSS
    
    cv.imwrite(f"images/captured/processed/processed ({captured_time}).jpg", processed_frame)
    cv.imwrite(f"images/captured/original/original ({captured_time}).jpg", original_frame)

    return processed_frame,balance_out,fabric_side,gusset_side,fabric_damage,blurred_otsu

