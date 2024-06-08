import os
import cv2 as cv
from skimage.feature import local_binary_pattern
import numpy as np



def openFile(count):
    file_path = "images\in\gusset ("+str(count)+").jpg"
    print(file_path)
    if not os.path.exists(file_path):
        print("Error: File '{}' not found.".format(file_path))
        exit()
    return(file_path)




def camera(source):
    _,original_frame=source.read()
    #cv.imshow('Original Image', original_frame)
    return original_frame



def preprocess_cpu(original_frame, c):
    threshold1 = 100
    threshold2 = 200

    grayscale_image = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)

    # Otsu's Binarization
    _, otsu_thresholded = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(otsu_thresholded, (5, 5), 0)

    # Apply Canny edge detection
    canny = cv.Canny(blurred_otsu, threshold1, threshold2)

    return blurred_otsu, canny

def preprocess_gpu(original_frame, gpu_original_frame, gaussian_filter, canny_edge_detector, c):
    gpu_original_frame.upload(original_frame)

    gpu_grayscale_image = cv.cuda.cvtColor(gpu_original_frame, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    gpu_blurred_image = gaussian_filter.apply(gpu_grayscale_image)

    # Otsu's Binarization
    otsu_threshold_value, gpu_otsu_thresholded = cv.cuda.threshold(gpu_blurred_image, 180, 255, 3)

    # Verify if Otsu threshold value is calculated correctly
    if otsu_threshold_value is None or gpu_otsu_thresholded.empty():
        print("Error: Otsu thresholding failed on GPU")
        return None, None

    gpu_blurred_otsu = gaussian_filter.apply(gpu_otsu_thresholded)

    # Apply Canny edge detection
    gpu_canny = canny_edge_detector.detect(gpu_blurred_otsu)

    blurred_otsu = gpu_blurred_otsu.download()
    canny = gpu_canny.download()

    return blurred_otsu, canny

def log(log_data):
    with open("log.txt", "a") as file:
        # Append data to the file
        file.write(log_data)

