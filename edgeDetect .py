import cv2 as cv
import numpy as np
import os

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
    cv.imshow('Original Image', original_frame)

    # Apply Laplacian edge detection
    laplace = cv.Laplacian(original_frame, cv.CV_64F)
    laplace = np.uint8(np.absolute(laplace))
    cv.imshow('Laplacian Edge', laplace)

    # Apply Canny edge detection
    canny = cv.Canny(original_frame, threshold1, threshold2)
    cv.imshow('Canny Edge', canny)

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
        cv.drawContours(frame_contours, [second_longest_contour], -1, (0, 0, 255), 2)

        x0, y0 = longest_contour[0][0]
        x1, y1 = longest_contour[1][0]
        x2, y2 = longest_contour[2][0]

        m=(y2-y0)/(x2-x0)
        mTan=-1/m
        xTan=x0+2
        yTan=int(mTan*(xTan-x0)+y0)
        
        cv.line(frame_contours, (x0,y0),  (xTan,yTan) , (255, 255, 255), 2)

        cv.putText(frame_contours, str(mTan), (x0+10,y0-10), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)

        #coord0=str(x0)+","+str(y0)
        #cv.putText(frame_contours, coord0, (x0+10,y0-10), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)

        #coord1=str(x1)+","+str(y1)
        #cv.putText(frame_contours, coord1, (x1+10,y1-10), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)

        
        cv.circle(frame_contours, (x0, y0),3, (0, 0, 0), -1)
        cv.circle(frame_contours, (x1, y1),3, (0, 0, 0), -1)    


        cv.imshow('Longest Edge', frame_contours)        
    else:
        cv.imshow('Longest Edge', original_frame)

    # Wait for 'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break

# Release resources
cv.destroyAllWindows()
