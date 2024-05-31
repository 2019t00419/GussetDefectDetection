import os
import cv2 as cv




def openFile(count):
    file_path = "images\in\gusset ("+str(count)+").jpg"
    print(file_path)
    if not os.path.exists(file_path):
        print("Error: File '{}' not found.".format(file_path))
        exit()
    return(file_path)




def camera(source):
    _,original_frame=source.read()
    cv.imshow('Original Image', original_frame)
    return original_frame



def preprocess(original_frame):
    threshold1=100
    threshold2=200

    original_frame = cv.resize(original_frame, (960, 1280))
    original_frame_resized = cv.resize(original_frame, (960, 1280))
    grayscale_image = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)

    # Otsu's Binarization
    _, otsu_thresholded = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(otsu_thresholded, (5, 5), 0)
    otsu_resized = cv.resize(blurred_otsu, (960, 1280))    
    
    cv.imshow('otsu_thresholded', otsu_resized)


    # Apply Canny edge detection
    canny = cv.Canny(blurred_otsu, threshold1, threshold2)
    canny_resized = cv.resize(canny, (960, 1280))

    cv.imshow('Canny Edge', canny_resized)
    return original_frame,original_frame_resized,blurred_otsu,canny