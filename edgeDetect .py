import cv2 as cv
import numpy as np
import os

camera= cv.VideoCapture(0)

# Check if the file exists
#file_path = 'gusset.jpg'
#if not os.path.exists(file_path):
    #print("Error: File '{}' not found.".format(file_path))
    #exit()

# Read the image
#original_frame = cv.imread(file_path)
threshold1=200
threshold2=400
while True:
    
    _,original_frame=camera.read()
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
        x2, y2 = longest_contour[2][0]


        m=(y2-y0)/(x2-x0)
        mTan=-1/m
        xTan=x1+2
        yTan=int(mTan*(xTan-x1)+y1)
        
        cv.line(frame_contours, (x1,y1),  (xTan,yTan) , (255, 255, 255), 2)

        cv.putText(frame_contours, "Slope of the tangent is "+str(mTan), (100,100), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)
        cv.putText(frame_contours, str(x1)+","+str(y1), (x1,y1), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)
        #cv.putText(frame_contours, str(xTan)+","+str(yTan), (xTan,yTan), cv.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 0), 2, cv.LINE_AA)
       
        cv.circle(frame_contours, (x1, y1),3, (0, 0, 0), -1)

        xa,ya=second_longest_contour[0][0]
        xUpLimit,yUpLimit=1000000,1000000
        xDownLimit,yDownLimit=-1000000,-1000000
        itr=0
        upLimitPoint=0

        for point in second_longest_contour:
            xTemp, yTemp = point[0] 
            itr=itr+1
            if(xTemp>x1):
                if(xTemp<xUpLimit):
                    xUpLimit=xTemp
                    yUpLimit=yTemp
                    upLimitPoint=itr
            elif(xTemp==x1):
                xa=xTemp
                ya=yTemp
                upLimitPoint=itr
        if(xa!=x1):
            xDownLimit,yDownLimit=second_longest_contour[upLimitPoint-1][0]

            #mNew=(yUpLimit-yDownLimit)/(xUpLimit-xDownLimit)
            #ya=int(mNew*(x1-xDownLimit)+yDownLimit)
            xa=x1
            
        
        #cv.putText(frame_contours, "x upper limit"+str(xUpLimit)+", x lower limit"+str(xDownLimit), (300, 300), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2, cv.LINE_AA)
        #print("ya is "+str(ya))

        
        cv.circle(frame_contours, (xUpLimit, yUpLimit), 3, (0, 0, 0), -1)
        cv.putText(frame_contours, "Upper Limit"+ str(xUpLimit)+","+str(yUpLimit), (xUpLimit, yUpLimit-20), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2, cv.LINE_AA)

        
        cv.circle(frame_contours, (xDownLimit, yDownLimit), 3, (0, 0, 0), -1)
        cv.putText(frame_contours, "Lower Limit"+str(xDownLimit)+","+str(yDownLimit), (xDownLimit, yDownLimit+20), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2, cv.LINE_AA)


        cv.circle(frame_contours, (xa, ya), 3, (0, 0, 0), -1)
        cv.putText(frame_contours, str(xa)+","+str(ya), (xa, ya), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2, cv.LINE_AA)


        cv.imshow('Longest Edge', frame_contours)        
    else:
        cv.imshow('Longest Edge', original_frame)

    # Wait for 'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break

# Release resources
cv.destroyAllWindows()
