import cv2 as cv
import numpy as np

def identify_edges(contours,sample_longest_contour,sample_second_longest_contour):
    
    max_length = 0
    second_max_length = 0
    longest_contour = None
    second_longest_contour = None
    gusset_side = None

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
    if longest_contour is not None:
        longest_contour_uncertainity = cv.matchShapes(longest_contour,sample_longest_contour,1,0.0)  
        if longest_contour_uncertainity < 0.2:
            gusset_side = "Front"
            if second_longest_contour is not None:
                second_longest_contour_uncertainity = cv.matchShapes(second_longest_contour,sample_second_longest_contour,1,0.0)
                if second_longest_contour_uncertainity < 0.2:
                    gusset_side = "Back"
                else:
                    gusset_side = "Front"
                    second_longest_contour = None
            else:
                gusset_side = "Front"
                second_longest_contour = None
        else:
            gusset_side = None
            longest_contour = None
    return(gusset_side,longest_contour,second_longest_contour)  

def identify_inner_edge(contours,sample_second_longest_contour):
    
    max_length = 0
    inner_edge = None

    for contour in contours:
        length = cv.arcLength(contour, closed=True)
        #print(str(length))
        if length > max_length:
            max_length = length  
            inner_edge = contour

    if inner_edge is not None:
        inner_edge_uncertainity = cv.matchShapes(inner_edge,sample_second_longest_contour,1,0.0)  
        print(f"inner_edge_uncertainity : {inner_edge_uncertainity}")
        if inner_edge_uncertainity > 0.6:
            inner_edge = None
    return(inner_edge)

def identify_outer_edge(contours,sample_longest_contour):
    
    max_length = 0
    outer_edge = None

    for contour in contours:
        length = cv.arcLength(contour, closed=True)
        #print(str(length))
        if length > max_length:
            max_length = length  
            outer_edge = contour

    if outer_edge is not None:
        outer_edge_uncertainity = cv.matchShapes(outer_edge,sample_longest_contour,1,0.0)  
        #print(f"outer_edge_uncertainity : {outer_edge_uncertainity}")
        if outer_edge_uncertainity > 0.2:
            outer_edge = None
    return(outer_edge)
    


def sampleContours(sample_path):
    max_length = 0
    second_max_length = 0
    longest_contour = None
    second_longest_contour = None
    relative_length1 = 0

    image = cv.imread(sample_path)
    
    
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)

    _, cpu_thresholded_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(cpu_thresholded_image, (5, 5), 0)
    canny = cv.Canny(blurred_otsu, 100, 200)
    
    # Find contours and draw the bounding box of the largest contour
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    #cv.imshow("canny",canny)

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
    
    detection_mask = np.zeros_like(grayscale_image)
    cv.drawContours(detection_mask, [second_longest_contour], -1, 255, 1)
    cv.drawContours(detection_mask, [longest_contour],  -1, 255, 1)
    
    detection_mask = cv.resize(detection_mask, (360,640)) 
    #cv.imshow("sample image",detection_mask)

    if longest_contour is not None and second_longest_contour is not None:
        print("\n\n\n\nSample image is successfull\n\n\n\n")
    else:
        print("\n\n\n\nSample image failed\n\n\n\n")
    return(longest_contour,second_longest_contour,image)
                     