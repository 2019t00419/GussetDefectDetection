
from contourID import identify_edges
import cv2 as cv
import numpy as np



def detect_gusset(contours,display_image,grayscale_image,x_margins,y_margins,frame_width,frame_height,capturedIn,canny):
    cx,cy=0,0
    box,longest_contour,second_longest_contour = None,None,None
    gussetIdentified = False
    captured = capturedIn
    
    if contours:
        longest_contour,second_longest_contour=identify_edges(contours)

        if longest_contour is not None:
            #cv.drawContours(display_image, [box], 0, (0, 0, 255), 2)

            (x, y), (MA, ma), angle = cv.fitEllipse(longest_contour)
            #cv.putText(display_image, f"Major axis length: {int(MA)}    Minor axis length: {int(ma)}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            
            # Create a mask for the largest contour
            mask = np.zeros_like(grayscale_image)
            cv.drawContours(mask, [longest_contour], -1, 255, thickness=cv.FILLED)
            cv.drawContours(canny, [longest_contour], -1, 0, 2)
            ret = cv.matchShapes(longest_contour,sampleContour(),1,0.0)

            confidence = (1-ret)*100
            if confidence <= 0:
                print(f"Gusset detection confidence is {0}%")
            else :
                print(f"Gusset detection confidence is {confidence}%")
                    
            if ret<0.2:
                if second_longest_contour is not None:
                    cv.drawContours(display_image, [second_longest_contour], -1, (255, 0, 255), 1)
                    cv.drawContours(display_image, [longest_contour], -1, (255, 0, 0), 1)

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
                        gussetIdentified = False
                    else :
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        cv.circle(display_image, (cx, cy), 5, (255, 0, 0), cv.FILLED)
                        gussetIdentified = True
            return gussetIdentified,cx,cy,box,longest_contour,second_longest_contour,display_image,grayscale_image,captured,ma,MA
        
        else:            
            return gussetIdentified,cx,cy,box,longest_contour,second_longest_contour,display_image,grayscale_image,captured,ma,MA




def rem(cx,cy,box,longest_contour,second_longest_contour,display_image,grayscale_image,canny):
    cv.drawContours(display_image, [box], 0, (0, 0, 255), 2)

    (x, y), (MA, ma), angle = cv.fitEllipse(longest_contour)
    #cv.putText(display_image, f"Major axis length: {int(MA)}    Minor axis length: {int(ma)}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    
    # Create a mask for the largest contour
    mask = np.zeros_like(grayscale_image)
    cv.drawContours(mask, [longest_contour], -1, 255, thickness=cv.FILLED)
    cv.drawContours(canny, [longest_contour], -1, 0, 2)
    ret = cv.matchShapes(longest_contour,sampleContour(),1,0.0)
    confidence = (1-ret)*100
    if confidence <= 0:
        print(f"Gusset detection confidence is {0}%")
    else :
        print(f"Gusset detection confidence is {confidence}%")
              
    if ret<0.5:
        if second_longest_contour is not None:
            cv.drawContours(display_image, [second_longest_contour], -1, (255, 0, 255), 1)
            cv.drawContours(display_image, [longest_contour], -1, (255, 0, 0), 1)
    return ma, MA


def sampleContour():
    image = cv.imread("Images/sample/sample (0).jpg")
    
    
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)

    _, cpu_thresholded_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(cpu_thresholded_image, (5, 5), 0)
    canny = cv.Canny(blurred_otsu, 100, 200)
    
    # Find contours and draw the bounding box of the largest contour
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.imshow("canny",canny)

    if contours:
        sample_contour = max(contours, key=cv.contourArea)
        return sample_contour