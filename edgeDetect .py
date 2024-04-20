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

        x0, y0 = longest_contour[0][0]
        x1, y1 = longest_contour[1][0]

        
        xa, ya = second_longest_contour[0][0]

        if(x1-x0 !=0):
            m_tangent = (-y1+y0)/(x1-x0)
            c_tangent = y0-(m_tangent*x0)
            eqn="y = "+ str(m_tangent)+"x + "+ str(c_tangent)
            distance = ((m_tangent*xa)-ya+c_tangent)/(np.sqrt((m_tangent*m_tangent)+1))
            x_approx0 = 250
            y_approx0 = int((m_tangent*x_approx0)+c_tangent)
            x_approx1 = 750
            y_approx1 = int((m_tangent*x_approx1)+c_tangent)

            m_normal = -1/m_tangent
            c_normal = y0-(m_normal*x0)


            
            cv.line(frame_contours, (x_approx0,y_approx0),  (x_approx1,y_approx1) , (255, 255, 255), 2)
        else:
            eqn="x = "+ str(x0)


        
        #Equation of the line
        cv.putText(frame_contours, "Equation of the outer contour at ("+ str(x0)+","+ str(y0)+ ") and ("+ str(x1)+","+ str(y1) +" is ", (100,100), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)
        cv.putText(frame_contours, eqn, (100,140), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)
        cv.putText(frame_contours, "Distance is ", (100,160), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)
        cv.putText(frame_contours, str(distance), (100,190), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)
        #cv.putText(frame_contours, str(x1)+","+str(y1), (x1,y1), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)
        #cv.putText(frame_contours, str(xTan)+","+str(yTan), (xTan,yTan), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)
       
        cv.circle(frame_contours, (x1, y1),3, (0, 0, 0), -1)
        cv.circle(frame_contours, (xa, ya),3, (0, 0, 0), -1)

        cv.imshow('Longest Edge', frame_contours)        
    else:
        cv.imshow('Longest Edge', original_frame)

    # Wait for 'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break

# Release resources
cv.destroyAllWindows()
