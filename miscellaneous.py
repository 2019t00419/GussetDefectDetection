import os
import cv2 as cv
import numpy as np
import time




def openFile(count):
    file_path = "images\in\gusset ("+str(count)+").jpg"
    print(file_path)
    if not os.path.exists(file_path):
        print("Error: File '{}' not found.".format(file_path))
        exit()
    return(file_path)




# Function to initialize webcam with given resolution
def initialize_cam(width, height, backend=cv.CAP_DSHOW):
    cap = cv.VideoCapture(0, backend)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    return cap



def preprocess(original_frame,c):
    threshold1=100
    threshold2=200

    #original_frame = cv.resize(original_frame, (960, 1280))
    original_frame_resized = cv.resize(original_frame, (960, 1280))
    grayscale_image = cv.cvtColor(original_frame_resized, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)

    # Otsu's Binarization
    _, otsu_thresholded = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(otsu_thresholded, (5, 5), 0)
    otsu_resized = cv.resize(blurred_otsu, (960, 1280))    
    
    #cv.imshow('otsu_thresholded', otsu_resized)


    # Apply Canny edge detection
    canny = cv.Canny(blurred_otsu, threshold1, threshold2)
    canny_resized = cv.resize(canny, (960, 1280))

    #cv.imshow('Canny Edge', canny)


    return original_frame,original_frame_resized,blurred_otsu,canny,blurred_image,grayscale_image

def preprocess_for_detection(image):
    
    display_image = image.copy()
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    detection_mask = np.zeros_like(grayscale_image)

    frame_height, frame_width = grayscale_image.shape
    detection_length = int(frame_width*0.95)
    detection_height = int(frame_height*0.95)
    x_margins = int((frame_width - detection_length) / 2)
    y_margins = int((frame_height - detection_height) / 2)


    # Create a rectangular mask
    cv.rectangle(detection_mask, (x_margins, y_margins), (frame_width - x_margins, frame_height - y_margins), 255, cv.FILLED)

    # Draw the rectangle on the display image for visualization
    cv.rectangle(display_image, (x_margins, y_margins), (frame_width - x_margins, frame_height - y_margins), (255, 255, 255), 2)

    # Apply the mask to the grayscale image
    masked_grayscale_image = cv.bitwise_and(grayscale_image, grayscale_image, mask=detection_mask)

    # Show the masked grayscale image
    # CPU operations
    blurred_image = cv.GaussianBlur(masked_grayscale_image, (5, 5), 0)
    _, cpu_thresholded_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(cpu_thresholded_image, (5, 5), 0)
    canny = cv.Canny(blurred_otsu, 100, 200)
    cv.imshow("canny",canny)
    
    # Find contours and draw the bounding box of the largest contour
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    return contours,display_image,grayscale_image,x_margins,y_margins,frame_width,frame_height,canny


def calculateFPS(cpu_times,end_cpu,start_cpu,last_update_time):
    update_interval = 1000  # Update FPS every second
    cpu_time = (end_cpu - start_cpu) * 1000
    cpu_times.append(cpu_time)
    #print("CPU time : " + str(cpu_time) + "ms")
    current_time = time.time()
    if current_time - last_update_time >= update_interval:
        avg_cpu_time = np.mean(cpu_times)
        avg_cpu_fps = 1000 / avg_cpu_time if avg_cpu_time > 0 else 0

        print("Average CPU FPS : " + str(avg_cpu_fps))
        cpu_times = []
        last_update_time = current_time
    return avg_cpu_fps