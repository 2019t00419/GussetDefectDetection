import cv2 as cv
import numpy as np
from scipy.spatial import KDTree



def checkBalanceOut(longest_contour, second_longest_contour, frame_contours,adhesiveWidthStr,printY):
    if (adhesiveWidthStr == "4mm"):
        adhesiveWidth = 4
    elif(adhesiveWidthStr == "6mm"):
        adhesiveWidth = 6
        
    tolerance = 1
    balance_out = False
    fontSize=5
    fontThickness = 3
    printY = 300
    lineSpace = 70
    
    # Convert contours to NumPy arrays for efficient computation
    longest_contour = np.array(longest_contour)
    second_longest_contour = np.array(second_longest_contour)
    
    # Get the coordinates from the contours
    longest_coords = longest_contour[:, 0]
    second_coords = second_longest_contour[:, 0]
    
    # Create a KDTree for the longest_contour
    kdtree = KDTree(longest_coords)
    
    # Find the minimum distances for all points in second_longest_contour using KDTree
    min_distances, nearest_indices = kdtree.query(second_coords)
    
    # Calculate the average minimum distance
    #avg_dist = np.mean(min_distances)
    #adhesiveWidth = pix_to_mm(avg_dist)

    # Variables for tracking
    itr_count = 0
    sum_distances = 0
    gap = 100

    # Variables for text display coordinates
    x_out_display, y_out_display = 0, 0
    x_in_display, y_in_display = 0, 0
    
    for i, inner_coordinates in enumerate(second_coords):
        min_dist = min_distances[i]
        
        sum_distances += min_dist
        itr_count += 1
        
        if itr_count > gap:
            avg_segment_dist = pix_to_mm(sum_distances / itr_count)
            sum_distances = 0
            itr_count = 0
            
            if ((avg_segment_dist > (adhesiveWidth + tolerance)) or (avg_segment_dist < (adhesiveWidth - tolerance))):
                balance_out = True
                color = (0, 0, 255)
            else:
                color = (0, 0, 0)
            
            cv.putText(frame_contours, str(round(avg_segment_dist, 2)), (x_out_display, y_out_display), cv.FONT_HERSHEY_PLAIN, 3.5, color, 3, cv.LINE_AA)
            cv.line(frame_contours, (x_out_display, y_out_display), (x_in_display, y_in_display), color, 2)
        
        elif itr_count == gap // 2:
            nearest_point_index = nearest_indices[i]
            x_out_display, y_out_display = longest_coords[nearest_point_index]
            x_in_display, y_in_display = inner_coordinates

    
    cv.putText(frame_contours, "Adhesive Width : " + str(adhesiveWidth)+ "mm", (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (255, 255, 255), fontThickness, cv.LINE_AA)
    printY += lineSpace
    cv.putText(frame_contours, "Tolerance : " + str(tolerance) + "mm", (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (255, 255, 255), fontThickness, cv.LINE_AA)
    printY += lineSpace


    return(balance_out,printY,frame_contours)


def check_fabric_damage(assisted_defects_image,frame_contours):
    fabric_damage_bool = False
    # Threshold for detecting dense clusters
    solidity_threshold = 0.6

    # Find contours in the binary mask
    defect_contours, _ = cv.findContours(assisted_defects_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw red bounding boxes on the original image for dense clusters only
    for defect_contour in defect_contours:
        # Calculate the area of the contour and its convex hull
        contour_area = cv.contourArea(defect_contour)
        if contour_area == 0:
            continue

        hull = cv.convexHull(defect_contour)
        hull_area = cv.contourArea(hull)

        # Calculate solidity (contour_area / hull_area)
        solidity = contour_area / hull_area

        # Check if the contour is dense enough based on solidity
        print(f"Solidity of potential fabric damages : {solidity}")
        if solidity >= solidity_threshold:
            fabric_damage_bool = True
            x, y, w, h = cv.boundingRect(defect_contour)  # Get bounding box coordinates
            cv.rectangle(frame_contours, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Draw red rectangle
            cv.drawContours(frame_contours, [defect_contour],  -1, (0,0,255), 3)

    return fabric_damage_bool,frame_contours


def checkPanelCutDamage(longest_contour,assisted_fabric_mask,frame_contours):
    solidity_threshold = 0.6
    panel_cut_damage_bool = False
    
    #isolate panel cut damage
    gusset_mask = np.ones_like(assisted_fabric_mask)
    cv.drawContours(gusset_mask, [longest_contour], -1, (255), cv.FILLED)   

    invert_gusset_mask  = cv.bitwise_not(gusset_mask)


    panel_cut_damage_mask = cv.bitwise_and(assisted_fabric_mask, assisted_fabric_mask, mask=invert_gusset_mask)



    blurred_panel_cut_damage_mask = cv.GaussianBlur(panel_cut_damage_mask, (5, 5), 0)
    canny_check = cv.Canny(blurred_panel_cut_damage_mask, 100, 200)
    

    defect_contours, _ = cv.findContours(canny_check, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Draw red bounding boxes on the original image for dense clusters only
    for defect_contour in defect_contours:
        # Calculate the area of the contour and its convex hull
        contour_area = cv.contourArea(defect_contour)
        if contour_area == 0:
            continue

        hull = cv.convexHull(defect_contour)
        hull_area = cv.contourArea(hull)

        # Calculate solidity (contour_area / hull_area)
        solidity = contour_area / hull_area

        # Check if the contour is dense enough based on solidity
        print(f"Solidity of potential panel cut damages : {solidity}")
        if solidity >= solidity_threshold:
            panel_cut_damage_bool = True
            x, y, w, h = cv.boundingRect(defect_contour)  # Get bounding box coordinates
            cv.rectangle(frame_contours, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Draw red rectangle
            cv.drawContours(frame_contours, [defect_contour],  -1, (0,0,255), 3)

    return panel_cut_damage_bool,frame_contours



def pix_to_mm(pix):
    convert_factor = 14.4
    mm = pix/convert_factor
    return mm





def checkGussetPosition(gusset_identified,original_frame,frame_contours,original_frame_resized,longest_contour,second_longest_contour):
    frame_height, frame_width, channels = original_frame.shape
    resolution_factor = int(((frame_height ** 2) + (frame_width ** 2)) ** 0.5)
    #print("Resolution of the image is : "+str((frame_height*frame_width)/1000000)+"MP")
    #print("Resolution factor is : "+str(resolution_factor))


    if longest_contour is not None:
        y_limit=frame_height+900
        x_bound,y_bound,w_bound,h_bound = cv.boundingRect(longest_contour)
        #print("Y limit = "+str(y_limit))
        #print("Frame edge = "+str(frame_height))
        #print("Bounding box edge = "+str(y_bound+h_bound))
        cv.line(frame_contours, (0, y_limit), (frame_width, y_limit), (0,0,255), 2)
        cv.line(original_frame_resized, (0, y_limit), (frame_width, y_limit), (0,0,255), 2)
        cv.line(original_frame, (0, y_limit), (frame_width, y_limit), (0,0,255), 2)
        if(y_bound+h_bound < y_limit):
            cv.rectangle(frame_contours,(x_bound,y_bound),(x_bound+w_bound,y_bound+h_bound),(0,255,0),2)
                
            #draw contours on to the frame
            cv.drawContours(frame_contours, [longest_contour], -1, (0, 255, 0), thickness=3)
            
            #complete the incomplete coordinates
            #longest_contour=fill_coordinates(longest_contour)
        else:
            longest_contour = None
    else:
        ##cv.imshow('Edges', original_frame)
        print("BalanceOut:Invalid contours")

    if second_longest_contour is not None: 
        #draw contours on to the frame   
        cv.drawContours(frame_contours, [second_longest_contour], -1, (0, 0, 255), thickness=3)
        #plot the coordinates
    return(longest_contour)

