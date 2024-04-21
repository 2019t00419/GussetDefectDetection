import cv2 as cv
import numpy as np
import os


#camera= cv.VideoCapture(0)

# Check if the file exists
file_path = 'gusset.jpg'
if not os.path.exists(file_path):
    print("Error: File '{}' not found.".format(file_path))
    exit()

# Read the image
original_frame = cv.imread(file_path)
threshold1=200
threshold2=400
while True:
    
    #_,original_frame=camera.read()
    #cv.imshow('Original Image', original_frame)

    # Apply Laplacian edge detection
    laplace = cv.Laplacian(original_frame, cv.CV_64F)
    laplace = np.uint8(np.absolute(laplace))
    #cv.imshow('Laplacian Edge', laplace)

    # Apply Canny edge detection
    canny = cv.Canny(original_frame, threshold1, threshold2)
    #cv.imshow('Canny Edge', canny)

    # Find contours
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Find the longest contour
    max_length = 0
    second_max_length = 0
    longest_contour = None
    second_longest_contour = None

    for contour in contours:
        length = cv.arcLength(contour, closed=True)
        if length > max_length:
            max_length = length  
            longest_contour = contour
        elif length > second_max_length:
            second_max_length = length
            second_longest_contour = contour
        
    # Highlight the longest edge
    frame_contours = original_frame.copy()

    if longest_contour is not None:

        cv.drawContours(frame_contours, [longest_contour], -1, (0, 255, 0), 2)
        #cv.line(frame_contours, (0,500),  (1000,500) , (255, 255, 255), 2)
        #cv.line(frame_contours, (500,0),  (500,1000) , (255, 255, 255), 2)
    if second_longest_contour is not None:    
        cv.drawContours(frame_contours, [second_longest_contour], -1, (0, 0, 255), 2)

    if longest_contour is not None and second_longest_contour is not None:    

        insert_index=0
        last_x, last_y = longest_contour[len(longest_contour) - 1][0]

        for coordinates in longest_contour:
            next_x,next_y=coordinates[0]
            if(next_x-last_x < -1 or next_x-last_x > 1) and (next_y - last_y ==0):
                if next_x > last_x:
                    new_x = next_x-1
                    new_y = last_y
                    while new_x > last_x:
                        longest_contour = np.insert(longest_contour, insert_index, [[new_x,new_y]], axis=0)
                        new_x = new_x-1
                                
                else:
                    new_x = next_x+1
                    new_y = next_y
                    while new_x < last_x:
                        longest_contour = np.insert(longest_contour, insert_index, [[new_x,new_y]], axis=0)
                        new_x = new_x+1 
                                    
            #vertical case
            elif(next_y-last_y < -1 or next_y-last_y > 1) and (next_x - last_x == 0):
                if next_y > last_y:
                    new_y = next_y-1
                    new_x = last_x
                    while new_y > last_y:
                        longest_contour = np.insert(longest_contour, insert_index, [[new_x,new_y]], axis=0)
                        new_y = new_y-1
                                
                else:
                    new_x = next_x
                    new_y = next_y+1
                    while new_y < last_y:
                        longest_contour = np.insert(longest_contour, insert_index, [[new_x,new_y]], axis=0)
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
                        longest_contour = np.insert(longest_contour, insert_index, [[new_x,new_y]], axis=0)
                        new_x = new_x-1
                                
                else:
                    new_x = next_x+1
                    new_y = next_y
                    while new_x < last_x:
                        new_y = (line_gradient*new_x)+line_intercept
                        longest_contour = np.insert(longest_contour, insert_index, [[new_x,new_y]], axis=0)
                        new_x = new_x+1                            
            #print(str(next_x)+","+str(next_y))
            #print(str(last_x)+","+str(last_y))
                
            insert_index=insert_index+1    
            last_y = next_y
            last_x = next_x
        for coordinates in longest_contour:
            x, y = coordinates[0]  # Extract x and y coordinates from the point
            cv.circle(frame_contours, (x, y), 2, (0, 0, 255), -1)

#second contour

        insert_index=0
        last_x, last_y = second_longest_contour[len(second_longest_contour) - 1][0]

        for coordinates in second_longest_contour:
            next_x,next_y=coordinates[0]
            if(next_x-last_x < -1 or next_x-last_x > 1) and (next_y - last_y ==0):
                if next_x > last_x:
                    new_x = next_x-1
                    new_y = last_y
                    while new_x > last_x:
                        second_longest_contour = np.insert(second_longest_contour, insert_index, [[new_x,new_y]], axis=0)
                        new_x = new_x-1
                                
                else:
                    new_x = next_x+1
                    new_y = next_y
                    while new_x < last_x:
                        second_longest_contour = np.insert(second_longest_contour, insert_index, [[new_x,new_y]], axis=0)
                        new_x = new_x+1 
                                    
            #vertical case
            elif(next_y-last_y < -1 or next_y-last_y > 1) and (next_x - last_x == 0):
                if next_y > last_y:
                    new_y = next_y-1
                    new_x = last_x
                    while new_y > last_y:
                        second_longest_contour = np.insert(second_longest_contour, insert_index, [[new_x,new_y]], axis=0)
                        new_y = new_y-1
                                
                else:
                    new_x = next_x
                    new_y = next_y+1
                    while new_y < last_y:
                        second_longest_contour = np.insert(second_longest_contour, insert_index, [[new_x,new_y]], axis=0)
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
                        second_longest_contour = np.insert(second_longest_contour, insert_index, [[new_x,new_y]], axis=0)
                        new_x = new_x-1
                                
                else:
                    new_x = next_x+1
                    new_y = next_y
                    while new_x < last_x:
                        new_y = (line_gradient*new_x)+line_intercept
                        second_longest_contour = np.insert(second_longest_contour, insert_index, [[new_x,new_y]], axis=0)
                        new_x = new_x+1                            
            #print(str(next_x)+","+str(next_y))
            #print(str(last_x)+","+str(last_y))
                
            insert_index=insert_index+1    
            last_y = next_y
            last_x = next_x
        for coordinates in second_longest_contour:
            x, y = coordinates[0]  # Extract x and y coordinates from the point
            cv.circle(frame_contours, (x, y), 2, (0, 255, 0), -1)


        cv.imshow('Longest Edge', frame_contours)        
    else:
        cv.imshow('Longest Edge', original_frame)
        print("Invalid contours")
    cv.imwrite("Output.jpg",frame_contours)
    # Wait for 'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break

# Release resources
cv.destroyAllWindows()
