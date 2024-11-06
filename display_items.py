import cv2 as cv
import numpy as np
        
def outputs(gusset_identified,gusset_side,longest_contour,second_longest_contour,frame_contours,original_frame,blurred_otsu,canny,count,fabric_damage,defect_contours):
    if gusset_identified:
        if fabric_damage:
            # Draw red bounding boxes on the original image
            for defect_contour in defect_contours:
                x, y, w, h = cv.boundingRect(defect_contour)  # Get bounding box coordinates
                cv.rectangle(frame_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangle

        if gusset_side == "Front":
            fabric_area = cv.contourArea(longest_contour)
            cv.drawContours(frame_contours, [longest_contour], -1, (0, 0, 255), thickness=3)
            cv.putText(frame_contours, "Fabric_area : "+str(fabric_area), (400, 650), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
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
            cv.putText(frame_contours, "Total area : "+str(total_area), (400, 625), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
            cv.putText(frame_contours, "Fabric_area : "+str(fabric_area), (400, 650), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
            cv.putText(frame_contours, "Adhesive area : "+str(adhesive_area), (400, 675), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
            #display(frame_contours,longest_contour)
            ##cv.imshow('Edges', frame_contours_resized) 
            
            #cv.imwrite("images/out/output/back/Output(back side) ("+str(count)+").jpg",frame_contours)
            #print("Defect count :"+str(defect_count)+"\t Non defect count :"+str(non_defect_count))
    else:
        ##cv.imshow('Edges', original_frame_resized)

        #cv.imwrite("images\out\output\Output ("+str(count)+").jpg",original_frame)
        print("Invalid contours")

    #cv.imwrite("images\out\otsu\otsu ("+str(count)+").jpg",blurred_otsu)
    #cv.imwrite("images\out\canny\canny ("+str(count)+").jpg",canny)
    return frame_contours




def thumbnail_ganeration(sample_longest_contour,sample_second_longest_contour,sample_image,colour,thickness):

    if (colour == "Bianco"):
        colourValue = (220,220,220)
    elif(colour == "Nero"):        
        colourValue = (0,0,0)
    elif(colour == "Skin"):        
        colourValue = (150,200,250)
    else:    
        colourValue = (0,0,255)

    if(thickness == "4mm"):
        thicknessValue = 4
    elif(thickness == "6mm"):
        thicknessValue = 6
    else:
        thicknessValue = 0

    display_image = np.zeros_like(sample_image)
    if(thicknessValue == 4):
        cv.drawContours(display_image, [sample_longest_contour], -1, (255,255,255),  cv.FILLED)
        cv.drawContours(display_image, [sample_second_longest_contour], -1, colourValue, cv.FILLED)
    if(thicknessValue == 6):
        cv.drawContours(display_image, [sample_longest_contour],  -1, (255,255,255), cv.FILLED)
        cv.drawContours(display_image, [sample_second_longest_contour], -1, colourValue, cv.FILLED)
        cv.drawContours(display_image, [sample_second_longest_contour], -1, (255,255,255), 20)
    elif(thicknessValue == 0):
        cv.drawContours(display_image, [sample_longest_contour],  -1, (0,0,255), cv.FILLED)
        cv.drawContours(display_image, [sample_second_longest_contour], -1, colourValue, cv.FILLED)

    input_image_width,input_image_height,_=display_image.shape
    resize_factor = input_image_width/360
    
    display_thumbnail = cv.resize(display_image, (int(input_image_height/resize_factor),int((input_image_width/input_image_height)*(input_image_height/resize_factor))))
    
    return display_thumbnail

