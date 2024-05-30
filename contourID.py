import cv2 as cv

def identify_edges(contours):
    
    max_length = 0
    second_max_length = 0
    longest_contour = None
    second_longest_contour = None
    relative_length0 = 0
    relative_length1 = 0

    for contour in contours:
            length = cv.arcLength(contour, closed=True)
            #print(str(length))
            if length > max_length:
                if max_length != 0:
                    relative_length1=(length-max_length)/max_length
                    if 0.05 < relative_length1 < 0.5:
                        second_max_length = max_length
                        second_longest_contour = longest_contour
                        #print("longest : "+str(max_length)+" second longest : "+str(second_max_length)+ " relative : "+str(relative_length1))
                else:
                    second_max_length = max_length
                    second_longest_contour = longest_contour
                max_length = length  
                longest_contour = contour

            elif length > second_max_length:
                if second_max_length != 0:
                    relative_length1=(max_length-length)/length
                    if 0.05 < relative_length1 < 0.5:
                        second_max_length = length
                        second_longest_contour = contour
                     
    return(longest_contour,second_longest_contour)  
