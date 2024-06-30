import cv2 as cv
        
def outputs(gusset_identified,gusset_side,longest_contour,second_longest_contour,frame_contours,original_frame,original_frame_resized,blurred_otsu,canny,count):
    if gusset_identified:
        if gusset_side == "Front":
            fabric_area = cv.contourArea(longest_contour)
            cv.drawContours(frame_contours, [longest_contour], -1, (0, 0, 255), thickness=3)
            cv.putText(frame_contours, "Fabric_area : "+str(fabric_area), (400, 650), cv.FONT_HERSHEY_PLAIN, 2, (66,245,245), 2, cv.LINE_AA)
            #display(frame_contours,longest_contour)
            #cv.imshow('Edges', frame_contours_resized) 
            
            cv.imwrite("images/out/output/front/Output(front side) ("+str(count)+").jpg",frame_contours)
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
            #cv.imshow('Edges', frame_contours_resized) 
            
            cv.imwrite("images/out/output/back/Output(back side) ("+str(count)+").jpg",frame_contours)
            #print("Defect count :"+str(defect_count)+"\t Non defect count :"+str(non_defect_count))
    else:
        #cv.imshow('Edges', original_frame_resized)

        cv.imwrite("images\out\output\Output ("+str(count)+").jpg",original_frame)
        print("Invalid contours")

    cv.imwrite("images\out\otsu\otsu ("+str(count)+").jpg",blurred_otsu)
    cv.imwrite("images\out\canny\canny ("+str(count)+").jpg",canny)
    return frame_contours