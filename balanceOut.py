import cv2 as cv
import numpy as np
from scipy.spatial import KDTree



def checkBalanceOut(longest_contour, second_longest_contour, frame_contours):

    thickness = 4    
    tolerance = 1
    balance_out = False
    
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
    #thickness = pix_to_mm(avg_dist)

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
            
            if ((avg_segment_dist > (thickness + tolerance)) or (avg_segment_dist < (thickness - tolerance))):
                balance_out = True
                color = (0, 0, 255)
            else:
                color = (0, 0, 0)
            
            cv.putText(frame_contours, str(round(avg_segment_dist, 2)), (x_out_display, y_out_display), cv.FONT_HERSHEY_PLAIN, 2, color, 2, cv.LINE_AA)
            cv.line(frame_contours, (x_out_display, y_out_display), (x_in_display, y_in_display), color, 2)
        
        elif itr_count == gap // 2:
            nearest_point_index = nearest_indices[i]
            x_out_display, y_out_display = longest_coords[nearest_point_index]
            x_in_display, y_in_display = inner_coordinates

    if balance_out:
        cv.putText(frame_contours, "Defective", (400, 500), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame_contours, "Balance Out", (400, 550), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame_contours, "Non-Defective", (400, 500), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(frame_contours, "Thickness : " + str(thickness), (400, 575), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame_contours, "Tolerance : " + str(tolerance * 100) + "%", (400, 600), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)


    return(balance_out)


def pix_to_mm(pix):
    convert_factor = 9
    mm = pix/convert_factor
    return mm