import cv2 as cv
import numpy as np
        
def outputs(gusset_identified,gusset_side,longest_contour,second_longest_contour,longest_contour_check,frame_contours,original_frame,blurred_otsu,canny,defects,printY):
    fontSize=5
    fontThickness = 3
    lineSpace = 70
    if gusset_identified:
        cv.putText(frame_contours, str(defects), (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (0,0,255), fontThickness, cv.LINE_AA)
        printY += lineSpace

        if gusset_side == "Front":
            fabric_area = cv.contourArea(longest_contour_check)
            cv.drawContours(frame_contours, [longest_contour_check], -1, (0, 0, 255), thickness=3)
            cv.putText(frame_contours, "Fabric_area : "+str(fabric_area), (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (66,245,245), fontThickness, cv.LINE_AA)
            printY += lineSpace
            #display(frame_contours,longest_contour)
            ##cv.imshow('Edges', frame_contours_resized) 
            
            #cv.imwrite("images/out/output/front/Output(front side) ("+str(count)+").jpg",frame_contours)
            #print("Defect count :"+str(defect_count)+"\t Non defect count :"+str(non_defect_count))
        
        elif gusset_side == "Back":
        
            total_area = cv.contourArea(longest_contour)
            fabric_area = cv.contourArea(second_longest_contour)
            adhesive_area = total_area - fabric_area
            
            cv.drawContours(frame_contours, [longest_contour], -1, (0, 0, 255), thickness=3)
            cv.drawContours(frame_contours, [second_longest_contour], -1, (255, 0, 0), thickness=3)
            cv.putText(frame_contours, "Total area : "+str(total_area), (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (66,245,245), fontThickness, cv.LINE_AA)
            printY += lineSpace
            cv.putText(frame_contours, "Fabric_area : "+str(fabric_area), (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (66,245,245), fontThickness, cv.LINE_AA)
            printY += lineSpace
            cv.putText(frame_contours, "Adhesive area : "+str(adhesive_area), (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (66,245,245), fontThickness, cv.LINE_AA)
            printY += lineSpace
            #display(frame_contours,longest_contour)
            ##cv.imshow('Edges', frame_contours_resized) 
            
            #cv.imwrite("images/out/output/back/Output(back side) ("+str(count)+").jpg",frame_contours)
            #print("Defect count :"+str(defect_count)+"\t Non defect count :"+str(non_defect_count))
        else:
            
            if longest_contour is not None:
                total_area = cv.contourArea(longest_contour)
                cv.drawContours(frame_contours, [longest_contour], -1, (0, 0, 255), thickness=3)
                cv.putText(frame_contours, "Total area : "+str(total_area), (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (66,245,245), fontThickness, cv.LINE_AA)
                printY += lineSpace
                x1, y1, w1, h1 = cv.boundingRect(longest_contour)  # Get bounding box coordinates
                cv.rectangle(frame_contours, (x1-10, y1-10), (x1 + w1+10, y1 + h1 +10), (0, 0, 255), 3)  # Draw red rectangle
                if second_longest_contour is not None:
                    fabric_area = cv.contourArea(second_longest_contour)
                    cv.drawContours(frame_contours, [second_longest_contour], -1, (255, 0, 0), thickness=3)
                    adhesive_area = total_area - fabric_area
                    cv.putText(frame_contours, "Fabric_area : "+str(fabric_area), (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (66,245,245), fontThickness, cv.LINE_AA)
                    printY += lineSpace
                    cv.putText(frame_contours, "Adhesive area : "+str(adhesive_area), (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (66,245,245), fontThickness, cv.LINE_AA)
                    printY += lineSpace
                    x2, y2, w2, h2 = cv.boundingRect(second_longest_contour)  # Get bounding box coordinates
                    cv.rectangle(frame_contours, (x2 -10, y2 -10), (x2 + w2 +10, y2 + h2 +10), (0, 0, 255), 3)  # Draw red rectangle
            if longest_contour_check is not None:
                total_area = cv.contourArea(longest_contour_check)
                cv.drawContours(frame_contours, [longest_contour_check], -1, (0, 0, 255), thickness=3)
                cv.putText(frame_contours, "Total area : "+str(total_area), (400, printY), cv.FONT_HERSHEY_PLAIN, fontSize, (66,245,245), fontThickness, cv.LINE_AA)
                printY += lineSpace
                x3, y3, w3, h3 = cv.boundingRect(longest_contour_check)  # Get bounding box coordinates
                cv.rectangle(frame_contours, (x3 -10, y3 -10), (x3 + w3 +10, y3 + h3 +10), (0, 0, 255), 3)  # Draw red rectangle

    else:
        ##cv.imshow('Edges', original_frame_resized)

        #cv.imwrite("images\out\output\Output ("+str(count)+").jpg",original_frame)
        print("Invalid contours")

    #cv.imwrite("images\out\otsu\otsu ("+str(count)+").jpg",blurred_otsu)
    #cv.imwrite("images\out\canny\canny ("+str(count)+").jpg",canny)
    return frame_contours




def thumbnail_ganeration(sample_longest_contour,sample_second_longest_contour,sample_image,colour,adhesiveWidth):

    if (colour == "Bianco"):
        colourValue = (220,220,220)
    elif(colour == "Nero"):        
        colourValue = (0,0,0)
    elif(colour == "Skin"):        
        colourValue = (150,200,250)
    else:    
        colourValue = (0,0,255)

    if(adhesiveWidth == "4mm"):
        adhesiveWidthValue = 4
    elif(adhesiveWidth == "6mm"):
        adhesiveWidthValue = 6
    else:
        adhesiveWidthValue = 0

    display_image = np.zeros_like(sample_image)
    if(adhesiveWidthValue == 4):
        cv.drawContours(display_image, [sample_longest_contour], -1, (255,255,255),  cv.FILLED)
        cv.drawContours(display_image, [sample_second_longest_contour], -1, colourValue, cv.FILLED)
    if(adhesiveWidthValue == 6):
        cv.drawContours(display_image, [sample_longest_contour],  -1, (255,255,255), cv.FILLED)
        cv.drawContours(display_image, [sample_second_longest_contour], -1, colourValue, cv.FILLED)
        cv.drawContours(display_image, [sample_second_longest_contour], -1, (255,255,255), 20)
    elif(adhesiveWidthValue == 0):
        cv.drawContours(display_image, [sample_longest_contour],  -1, (0,0,255), cv.FILLED)
        cv.drawContours(display_image, [sample_second_longest_contour], -1, colourValue, cv.FILLED)

    input_image_width,input_image_height,_=display_image.shape
    resize_factor = input_image_width/360
    
    display_thumbnail = cv.resize(display_image, (int(input_image_height/resize_factor),int((input_image_width/input_image_height)*(input_image_height/resize_factor))))
    
    return display_thumbnail

