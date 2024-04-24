import cv2 as cv
import numpy as np

def fill_coordinates(contour_array):
    insert_index=0
    last_x, last_y = contour_array[len(contour_array) - 1][0]

    for coordinates in contour_array:
        next_x,next_y=coordinates[0]
        contour_array = np.insert(contour_array, insert_index, [[next_x,next_y]], axis=0)
        if(next_x-last_x < -1 or next_x-last_x > 1) and (next_y - last_y ==0):
            if next_x > last_x:
                new_x = next_x-1
                new_y = last_y
                while new_x > last_x:
                    contour_array = np.insert(contour_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x-1
                            
            else:
                new_x = next_x+1
                new_y = next_y
                while new_x < last_x:
                    contour_array = np.insert(contour_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x+1 
                                
        #vertical case
        elif(next_y-last_y < -1 or next_y-last_y > 1) and (next_x - last_x == 0):
            if next_y > last_y:
                new_y = next_y-1
                new_x = last_x
                while new_y > last_y:
                    contour_array = np.insert(contour_array, insert_index, [[new_x,new_y]], axis=0)
                    new_y = new_y-1
                            
            else:
                new_x = next_x
                new_y = next_y+1
                while new_y < last_y:
                    contour_array = np.insert(contour_array, insert_index, [[new_x,new_y]], axis=0)
                    new_y = new_y+1
                            
        #angled case    
        elif(next_x-last_x < -1 or next_x-last_x > 1) and (next_y-last_y < -1 or next_y-last_y > 1):

            #print(str(next_x)+","+str(next_y))
            #print(str(last_x)+","+str(last_y))
            line_gradient = ( (next_y - last_y)/(next_x-last_x) )
            line_intercept = ( next_y-(line_gradient*next_x))

            if next_x > last_x:
                    
                new_y = next_y
                new_x = next_x-1
                while new_x > last_x:
                    new_y = (line_gradient*new_x)+line_intercept
                    contour_array = np.insert(contour_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x-1
                            
            else:
                new_x = next_x+1
                new_y = next_y
                while new_x < last_x:
                    new_y = (line_gradient*new_x)+line_intercept
                    contour_array = np.insert(contour_array, insert_index, [[new_x,new_y]], axis=0)
                    new_x = new_x+1                            
        #print(str(next_x)+","+str(next_y))
        #print(str(last_x)+","+str(last_y))
            
        insert_index=insert_index+1    
        last_y = next_y
        last_x = next_x
    return contour_array



def measure_distance(longest_contour,second_longest_contour,frame_contours):
    tolerance=0.15
    thickness=32
    defective = False
    count=0
    print(len(second_longest_contour))
    for ref_point in range(0, int((len(second_longest_contour)-1)/2), 100):
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






def measure_distance_(longest_contour,second_longest_contour,frame_contours):
    tolerance=0.15
    thickness=32
    defective = False
    coord_index = 0
    ref_index = 0
    error_factor = 0.5
    
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
    predicted_distance = min_dist
    
    cv.putText(frame_contours,("predicted_distance : "+str(predicted_distance)) , (400,600), cv.FONT_HERSHEY_PLAIN, 1.5 , (255, 255, 255), 2, cv.LINE_AA)
    
    #Find nearest points on outer edge for each inner edge point
    distance = 0.0

    # Iterate through the contour points
    #print(len(longest_contour))
    #print(len(second_longest_contour))       
    #print(longest_contour[len(longest_contour)-1][0])
    min_index=ref_index
    count=0
    for inner_coordinates in second_longest_contour:
        min_dist=np.linalg.norm(inner_coordinates[0]-longest_contour[ref_index][0])
        for i in range(-200,200,1):
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
        cv.line(frame_contours,(inner_coordinates[0]),(longest_contour[min_index][0]),(255,0,0),2)
        ref_index=min_index
        count=count+1
    #print("Starting point : "+str(ref_index) +" : "+ str(longest_contour[ref_index][0]))
    
    return
