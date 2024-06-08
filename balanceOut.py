import cv2 as cv
import numpy as np
from scipy.spatial import KDTree



def measure_distance_KDTree(longest_contour, second_longest_contour, frame_contours):
    
    tolerance = 0.25
    defective = False
    
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
    avg_dist = np.mean(min_distances)
    thickness = pix_to_mm(avg_dist)

    # Variables for tracking
    itr_count = 0
    sum_distances = 0
    gap = 50

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
            
            if ((avg_segment_dist > thickness * (1 + tolerance)) or (avg_segment_dist < thickness * (1 - tolerance))):
                defective = True
                color = (0, 0, 255)
            else:
                color = (0, 0, 0)
            
            cv.putText(frame_contours, str(round(avg_segment_dist, 2)), (x_out_display, y_out_display), cv.FONT_HERSHEY_PLAIN, 2, color, 2, cv.LINE_AA)
            cv.line(frame_contours, (x_out_display, y_out_display), (x_in_display, y_in_display), color, 2)
        
        elif itr_count == gap // 2:
            nearest_point_index = nearest_indices[i]
            x_out_display, y_out_display = longest_coords[nearest_point_index]
            x_in_display, y_in_display = inner_coordinates

    if defective:
        cv.putText(frame_contours, "Defective", (400, 500), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame_contours, "Balance Out", (400, 550), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame_contours, "Non-Defective", (400, 500), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(frame_contours, "Thickness : " + str(thickness), (400, 575), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame_contours, "Tolerance : " + str(tolerance * 100) + "%", (400, 600), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)


    return(defective)




def checkBalanceOut(original_frame,frame_contours,longest_contour,second_longest_contour):
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
        cv.line(original_frame, (0, y_limit), (frame_width, y_limit), (0,0,255), 2)
        if(y_bound+h_bound < y_limit):
            cv.rectangle(frame_contours,(x_bound,y_bound),(x_bound+w_bound,y_bound+h_bound),(0,255,0),2)
                
            #draw contours on to the frame
            cv.drawContours(frame_contours, [longest_contour], -1, (0, 255, 0), thickness=cv.FILLED)
            
            #complete the incomplete coordinates
            #longest_contour=fill_coordinates(longest_contour)
        else:
            longest_contour = None
    else:
        #cv.imshow('Edges', original_frame)
        print("Invalid contours")

    if second_longest_contour is not None: 
        #draw contours on to the frame   
        cv.drawContours(frame_contours, [second_longest_contour], -1, (0, 0, 255), thickness=cv.FILLED)
        #complete the incomplete coordinates
        #second_longest_contour=fill_coordinates(second_longest_contour)
        #plot the coordinates
    return(longest_contour)


        
def outputs(longest_contour,second_longest_contour,frame_contours,original_frame,blurred_otsu,canny,count):
    if second_longest_contour is not None and longest_contour is not None:

        measure_distance_KDTree(longest_contour,second_longest_contour,frame_contours)
        #if measure_distance_KDTree(longest_contour,second_longest_contour,frame_contours):
        #    defect_count=defect_count+1
        #else :
        #    non_defect_count=non_defect_count+1
        
        total_area = cv.contourArea(longest_contour)
        fabric_area = cv.contourArea(second_longest_contour)
        adhesive_area = total_area - fabric_area
        cv.putText(frame_contours, "Total area : "+str(total_area), (400, 625), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
        cv.putText(frame_contours, "Fabric_area : "+str(fabric_area), (400, 650), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
        cv.putText(frame_contours, "Adhesive area : "+str(adhesive_area), (400, 675), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
        #display(frame_contours,longest_contour)
        
        frame_contours_resized = cv.resize(frame_contours, (960, 1280))
        #cv.imshow('Edges', frame_contours_resized) 
        
        cv.imwrite("images\out\output\Output ("+str(count)+").jpg",frame_contours)
        #print("Defect count :"+str(defect_count)+"\t Non defect count :"+str(non_defect_count))
    else:
        #cv.imshow('Edges', original_frame_resized)

        cv.imwrite("images\out\output\Output ("+str(count)+").jpg",original_frame)
        print("Invalid contours")

    cv.imwrite("images\out\otsu\otsu ("+str(count)+").jpg",blurred_otsu)
    cv.imwrite("images\out\canny\canny ("+str(count)+").jpg",canny)
    return frame_contours

def pix_to_mm(pix):
    convert_factor = 9
    mm = pix/convert_factor
    return mm