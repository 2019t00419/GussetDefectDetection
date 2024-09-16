
from contourID import identify_outer_edge
import cv2 as cv
import numpy as np



def detect_gusset(contours,display_image,grayscale_image,x_margins,y_margins,frame_width,frame_height,capturedIn,canny,sample_longest_contour,sample_second_longest_contour):
    cx,cy=0,0
    MA,ma = 0,0
    box,longest_contour= None,None
    gusset_detected = False
    captured = capturedIn
    confidence = 0
    
    if contours:
        longest_contour=identify_outer_edge(contours,sample_longest_contour)
        #print(f"sample_longest_contour = {sample_longest_contour}")

        if longest_contour is not None:
            #cv.drawContours(display_image, [box], 0, (0, 0, 255), 2)

            (x, y), (MA, ma), angle = cv.fitEllipse(longest_contour)
            #cv.putText(display_image, f"Major axis length: {int(MA)}    Minor axis length: {int(ma)}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

            # Create a mask for the largest contour
            ret = cv.matchShapes(longest_contour,sample_longest_contour,1,0.0)

            confidence = (1-ret)*100
            if confidence <= 0:
                confidence = 0
            #print(f"Gusset detection confidence is {confidence}%")
                    
            if ret<0.2:

                x, y, w, h = cv.boundingRect(longest_contour)
                cv.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if (x < x_margins) or (y < y_margins) or ((x+w) > (frame_width-x_margins)) or ((y+h) > (frame_height-y_margins)):
                    captured = False
                else:
                    rect = cv.minAreaRect(longest_contour)
                    box = cv.boxPoints(rect)
                    box = np.int0(box)

                    M = cv.moments(box)
                    if M['m00'] == 0:
                        gusset_detected = False
                    else :
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        cv.circle(display_image, (cx, cy), 5, (255, 0, 0), cv.FILLED)
                        gusset_detected = True
            return gusset_detected,cx,cy,box,longest_contour,display_image,grayscale_image,captured,ma,MA,confidence
        
        else:       
            #print(f"Longest contour is not available")     
            return gusset_detected,cx,cy,box,longest_contour,display_image,grayscale_image,captured,ma,MA,confidence

