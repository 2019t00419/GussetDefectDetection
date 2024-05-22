import cv2 as cv
import numpy as np
import time
from scipy.spatial import KDTree

def fill_coordinates(contour_array):
    insert_index=0
    last_x, last_y = contour_array[len(contour_array) - 1][0]
    new_array =contour_array
    for coordinates in contour_array:
        next_x,next_y=coordinates[0]
        if(next_x-last_x < -1 or next_x-last_x > 1) and (next_y - last_y ==0):
            if next_x > last_x:
                new_x = next_x-1
                new_y = last_y
                count=0
                while new_x > last_x:
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x-1
                    count=count+1
                insert_index=insert_index+count    
                if insert_index>len(new_array)-1:
                    break                        
                            
            else:
                new_x = next_x+1
                new_y = next_y
                count=0
                while new_x < last_x:
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x+1 
                    count=count+1
                insert_index=insert_index+count     
                if insert_index>len(new_array)-1:
                    break                                    
        #vertical case
        elif(next_y-last_y < -1 or next_y-last_y > 1) and (next_x - last_x == 0):
            if next_y > last_y:
                new_y = next_y-1
                new_x = last_x
                count=0
                while new_y > last_y:
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_y = new_y-1
                    count=count+1
                insert_index=insert_index+count     
                if insert_index>len(new_array)-1:
                    break                                
            else:
                new_x = next_x
                new_y = next_y+1
                count=0
                while new_y < last_y:
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_y = new_y+1
                    count=count+1
                insert_index=insert_index+count  
                if insert_index>len(new_array)-1:
                    break                       
                            
        #angled case    
        elif(next_x-last_x < -1 or next_x-last_x > 1) and (next_y-last_y < -1 or next_y-last_y > 1):

            #print(str(next_x)+","+str(next_y))
            #print(str(last_x)+","+str(last_y))
            line_gradient = ( (next_y - last_y)/(next_x-last_x) )
            line_intercept = ( next_y-(line_gradient*next_x))

            if next_x > last_x:
                    
                new_y = next_y
                new_x = next_x-1
                count=0
                while new_x > last_x:
                    new_y = (line_gradient*new_x)+line_intercept
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x-1
                    count=count+1
                insert_index=insert_index+count  
                if insert_index>len(new_array)-1:
                    break                       
                            
            else:
                new_x = next_x+1
                new_y = next_y
                count=0
                while new_x < last_x:
                    new_y = (line_gradient*new_x)+line_intercept
                    new_array = np.insert(new_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x+1  
                    count=count+1  
                insert_index=insert_index+count 
                if insert_index>len(new_array)-1:
                    break                        
        #print(str(next_x)+","+str(next_y))
        #print(str(last_x)+","+str(last_y))
            
        insert_index=insert_index+1    
        last_y = next_y
        last_x = next_x
    return new_array



def measure_distance_KDTree(longest_contour, second_longest_contour, frame_contours):
    start_time = time.time()  # Start time
    
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
    thickness = avg_dist

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
            avg_segment_dist = sum_distances / itr_count
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

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Time taken to complete the function: {elapsed_time:.4f} seconds")

    return


def measure_distance_my(longest_contour,second_longest_contour,frame_contours):
    start_time = time.time()  # Start time
    tolerance=0.25
    thickness=0
    defective = False
    coord_index = 0
    ref_index = 0
    
    #set the initial minimum distance as the distance between first two points in the edges
    min_dist=np.linalg.norm(longest_contour[0][0] - second_longest_contour[0][0])

    #searching for the shortest distance and the point for the first point of outer edge
    for outer_coordinates in longest_contour:
        #looping through all points of outer edge to find the minimum distance for thr first point of the inner edge
        dist = np.linalg.norm(second_longest_contour[0][0] - outer_coordinates[0])
        if dist<min_dist:
            min_dist=dist
            #this is the index of the starting coordinates for the outer edge
            ref_index = coord_index
        coord_index=coord_index+1
    #This is the predicted minimum distance for reference
    thickness = min_dist
    
    #cv.putText(frame_contours,("predicted_distance : "+str(predicted_distance)) , (400,600), cv.FONT_HERSHEY_PLAIN, 1.5 , (255, 255, 255), 2, cv.LINE_AA)
    
    #Find nearest points on outer edge for each inner edge point

    # Iterate through the contour points
    #print(len(longest_contour))
    #print(len(second_longest_contour))       
    #print(longest_contour[len(longest_contour)-1][0])
    min_index=ref_index
    itr_count=0
    x_out_display,y_out_display =0,0
    x_in_display,y_in_display =0,0
    sum=0

    for inner_coordinates in second_longest_contour:
        min_dist=np.linalg.norm(inner_coordinates[0]-longest_contour[ref_index][0])
        for i in range(-10,10,1):
            test_index=ref_index+i
            if(test_index>(len(longest_contour)-1)):
                test_index = test_index - (len(longest_contour))
            elif(test_index<0):
                test_index = (len(longest_contour))+test_index
            dist= np.linalg.norm(inner_coordinates[0]-longest_contour[test_index][0])
            if dist<min_dist:
                min_dist=dist
                min_index=test_index
            #print(str(count)+" Searching for "+str(inner_coordinates[0])+" test_index : "+str(test_index)+" Distance : "+str(dist)+" Current min : "+str(min_dist)+" Current min index : "+str(min_index))  
        
        #print(str(inner_coordinates[0])+str(longest_contour[min_index][0])+str(min_dist)+" Min_index : "+str(min_index))        
        #cv.line(frame_contours,(inner_coordinates[0]),(longest_contour[min_index][0]),(255,0,0),2)
        ref_index=min_index
        itr_count=itr_count+1
        sum=sum+min_dist
        gap =50
        if itr_count>gap:
            avg=sum/itr_count            
            sum=0
            itr_count=0
            if ((avg>thickness*(1+tolerance)) or (avg<thickness*(1-tolerance)) ) :
                defective = True
                color=(0,0,255)         
            else:
                color=(0,0,0)                   
            cv.putText(frame_contours,str(round(avg,2)) , (x_out_display,y_out_display), cv.FONT_HERSHEY_PLAIN, 2 , color, 2, cv.LINE_AA)
            cv.line(frame_contours,(x_out_display,y_out_display),(x_in_display,y_in_display),color,2)
        elif itr_count==(gap/2):
            x_out_display,y_out_display = (longest_contour[min_index][0])
            x_in_display,y_in_display = (inner_coordinates[0])

    #print("Starting point : "+str(ref_index) +" : "+ str(longest_contour[ref_index][0]))
    
    if(defective):
        cv.putText(frame_contours,("Defective") , (400,500), cv.FONT_HERSHEY_PLAIN, 2 , (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame_contours,("Balance Out") , (400,550), cv.FONT_HERSHEY_PLAIN, 2 , (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame_contours,("Non-Defective") , (400,500), cv.FONT_HERSHEY_PLAIN, 2 , (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(frame_contours,("Thickness : "+ str(thickness)) , (400,575), cv.FONT_HERSHEY_PLAIN, 1.5 , (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame_contours,("Tolerance : "+str(tolerance*100)+"%") , (400,600), cv.FONT_HERSHEY_PLAIN, 1.5 , (255, 255, 255), 2, cv.LINE_AA)

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Time taken to complete the function: {elapsed_time:.4f} seconds")

    return


def display(frame,cont) :
    for i in range(0,len(cont)-1,50):
        x,y=cont[i][0]
        cv.putText(frame,str(i),(x,y), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)
    

def measure_distance_(longest_contour,second_longest_contour,frame_contours):
    tolerance=0.5
    thickness=80
    defective = False
    count=0
    #print(len(second_longest_contour))
    for ref_point in range(0, int((len(second_longest_contour)-1)), 100):
        #print(ref_point)
        min_dist=np.linalg.norm(longest_contour[ref_point][0] - second_longest_contour[ref_point][0])
        x_outer,y_outer = longest_contour[ref_point][0]
        x_inner,y_inner = second_longest_contour[ref_point][0]
        for coordinates in second_longest_contour:
            dist = np.linalg.norm(longest_contour[ref_point][0] - coordinates[0])
            if dist<min_dist:
                min_dist=dist
                x_inner,y_inner = coordinates[0]
        #print(min_dist)
        #print(str(x_inner)+","+str(y_inner))
        #print(str(x_outer)+","+str(y_outer))
        if ((min_dist>thickness*(1+tolerance)) or (min_dist<thickness*(1-tolerance)) ) :
            defective = True
            color=(0,0,255)
        else:
            color=(0,0,0)
        cv.line(frame_contours,(x_outer,y_outer),(x_inner,y_inner),color,2)
        cv.putText(frame_contours,(str(ref_point)+": d="+str(min_dist)) , (x_outer,y_outer), cv.FONT_HERSHEY_PLAIN, 1.5 , color, 2, cv.LINE_AA)
        count=count+1
        
    if(defective):
        cv.putText(frame_contours,("Defective") , (400,500), cv.FONT_HERSHEY_PLAIN, 2 , (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame_contours,("Balance Out") , (400,550), cv.FONT_HERSHEY_PLAIN, 2 , (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame_contours,("Non-Defective") , (400,500), cv.FONT_HERSHEY_PLAIN, 2 , (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(frame_contours,("Thickness : "+ str(thickness)) , (400,575), cv.FONT_HERSHEY_PLAIN, 1.5 , (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame_contours,("Tolerance : "+str(tolerance)+"%") , (400,600), cv.FONT_HERSHEY_PLAIN, 1.5 , (255, 255, 255), 2, cv.LINE_AA)
    return



def measure_distance_optimized(longest_contour, second_longest_contour, frame_contours):
    start_time = time.time()  # Start time
    
    tolerance = 0.25
    defective = False
    
    # Convert contours to NumPy arrays for efficient computation
    longest_contour = np.array(longest_contour)
    second_longest_contour = np.array(second_longest_contour)
    
    # Initialize the minimum distance with the first point calculation
    min_dist = np.linalg.norm(longest_contour[0][0] - second_longest_contour[0][0])
    
    # Find the starting reference index for the shortest distance
    dist_array = np.linalg.norm(longest_contour[:, 0] - second_longest_contour[0][0], axis=1)
    ref_index = np.argmin(dist_array)
    thickness = dist_array[ref_index]

    # Variables for tracking
    min_index = ref_index
    itr_count = 0
    sum_distances = 0
    gap = 50

    # Variables for text display coordinates
    x_out_display, y_out_display = 0, 0
    x_in_display, y_in_display = 0, 0
    
    for inner_coordinates in second_longest_contour:
        # Calculate the distances for a range around the reference index
        test_indices = (np.arange(ref_index-10, ref_index+10) % len(longest_contour))
        test_points = longest_contour[test_indices][:, 0]
        dists = np.linalg.norm(test_points - inner_coordinates[0], axis=1)
        
        min_dist = np.min(dists)
        min_index = test_indices[np.argmin(dists)]
        
        ref_index = min_index
        itr_count += 1
        sum_distances += min_dist
        
        if itr_count > gap:
            avg_dist = sum_distances / itr_count
            sum_distances = 0
            itr_count = 0
            
            if ((avg_dist > thickness * (1 + tolerance)) or (avg_dist < thickness * (1 - tolerance))):
                defective = True
                color = (0, 0, 255)
            else:
                color = (0, 0, 0)
            
            cv.putText(frame_contours, str(round(avg_dist, 2)), (x_out_display, y_out_display), cv.FONT_HERSHEY_PLAIN, 2, color, 2, cv.LINE_AA)
            cv.line(frame_contours, (x_out_display, y_out_display), (x_in_display, y_in_display), color, 2)
        
        elif itr_count == gap // 2:
            x_out_display, y_out_display = longest_contour[min_index][0]
            x_in_display, y_in_display = inner_coordinates[0]

    if defective:
        cv.putText(frame_contours, "Defective", (400, 500), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame_contours, "Balance Out", (400, 550), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame_contours, "Non-Defective", (400, 500), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(frame_contours, "Thickness : " + str(thickness), (400, 575), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame_contours, "Tolerance : " + str(tolerance * 100) + "%", (400, 600), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Time taken to complete the function: {elapsed_time:.4f} seconds")

    return



def measure_distance_compare_mean(longest_contour, second_longest_contour, frame_contours):
    start_time = time.time()  # Start time
    
    tolerance = 0.25
    defective = False
    
    # Convert contours to NumPy arrays for efficient computation
    longest_contour = np.array(longest_contour)
    second_longest_contour = np.array(second_longest_contour)
    
    # Calculate distances from all points in the second_longest_contour to all points in the longest_contour
    distances = np.zeros((len(second_longest_contour), len(longest_contour)))
    for i, inner_coordinates in enumerate(second_longest_contour):
        distances[i] = np.linalg.norm(longest_contour[:, 0] - inner_coordinates[0], axis=1)
    
    # Find the minimum distance for each point in the second_longest_contour
    min_distances = np.min(distances, axis=1)
    
    # Calculate the average of these minimum distances
    avg_dist = np.mean(min_distances)
    
    # Initial thickness is based on the average minimum distance
    thickness = avg_dist

    # Variables for tracking
    itr_count = 0
    sum_distances = 0
    gap = 50

    # Variables for text display coordinates
    x_out_display, y_out_display = 0, 0
    x_in_display, y_in_display = 0, 0
    
    for i, inner_coordinates in enumerate(second_longest_contour):
        min_dist = min_distances[i]
        
        sum_distances += min_dist
        itr_count += 1
        
        if itr_count > gap:
            avg_segment_dist = sum_distances / itr_count
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
            nearest_point_index = np.argmin(distances[i])
            x_out_display, y_out_display = longest_contour[nearest_point_index][0]
            x_in_display, y_in_display = inner_coordinates[0]

    if defective:
        cv.putText(frame_contours, "Defective", (400, 500), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame_contours, "Balance Out", (400, 550), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame_contours, "Non-Defective", (400, 500), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(frame_contours, "Thickness : " + str(thickness), (400, 575), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame_contours, "Tolerance : " + str(tolerance * 100) + "%", (400, 600), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Time taken to complete the function: {elapsed_time:.4f} seconds")

    return



def measure_distance_fast_compare_mean(longest_contour, second_longest_contour, frame_contours):
    start_time = time.time()  # Start time
    
    tolerance = 0.25
    defective = False
    
    # Convert contours to NumPy arrays for efficient computation
    longest_contour = np.array(longest_contour)
    second_longest_contour = np.array(second_longest_contour)
    
    # Get the coordinates from the contours
    longest_coords = longest_contour[:, 0]
    second_coords = second_longest_contour[:, 0]
    
    # Calculate the minimum distances for all points in second_longest_contour
    dist_matrix = np.linalg.norm(longest_coords[:, np.newaxis] - second_coords, axis=2)
    min_distances = np.min(dist_matrix, axis=0)
    
    # Calculate the average minimum distance
    avg_dist = np.mean(min_distances)
    thickness = avg_dist

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
            avg_segment_dist = sum_distances / itr_count
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
            nearest_point_index = np.argmin(dist_matrix[:, i])
            x_out_display, y_out_display = longest_coords[nearest_point_index]
            x_in_display, y_in_display = inner_coordinates

    if defective:
        cv.putText(frame_contours, "Defective", (400, 500), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame_contours, "Balance Out", (400, 550), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame_contours, "Non-Defective", (400, 500), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(frame_contours, "Thickness : " + str(thickness), (400, 575), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame_contours, "Tolerance : " + str(tolerance * 100) + "%", (400, 600), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Time taken to complete the function: {elapsed_time:.4f} seconds")

    return

