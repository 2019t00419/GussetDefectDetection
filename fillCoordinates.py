import cv2 as cv
import numpy as np

def fill_coordinates(contour_array):
    insert_index=0
    last_x, last_y = contour_array[len(contour_array) - 1][0]

    for coordinates in contour_array:
        next_x,next_y=coordinates[0]
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
    for ref_point in range(0, int(len(second_longest_contour)/2), 100):
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
        cv.putText(frame_contours,("d="+str(min_dist)) , (x_outer,y_outer), cv.FONT_HERSHEY_PLAIN, 1.5 , color, 2, cv.LINE_AA)
        
    if(defective):
        cv.putText(frame_contours,("Defective") , (400,500), cv.FONT_HERSHEY_PLAIN, 2 , (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame_contours,("Balance Out") , (400,550), cv.FONT_HERSHEY_PLAIN, 2 , (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame_contours,("Non-Defective") , (400,500), cv.FONT_HERSHEY_PLAIN, 2 , (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(frame_contours,("Thickness : "+ str(thickness)) , (400,575), cv.FONT_HERSHEY_PLAIN, 1.5 , (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame_contours,("Tolerance : "+str(tolerance)+"%") , (400,600), cv.FONT_HERSHEY_PLAIN, 1.5 , (255, 255, 255), 2, cv.LINE_AA)
    return