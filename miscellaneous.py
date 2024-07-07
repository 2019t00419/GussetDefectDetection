import os
import cv2 as cv
import numpy as np
import time
from contourID import identify_inner_edge




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



def preprocess(original_frame,style,sample_longest_contour,sample_second_longest_contour):
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

    #cv.imshow('Canny Edge', canny)
    if style == "Light":
        original_frame_for_hsv = cv.resize(original_frame, (720, 1280))
        hsv = cv.cvtColor(original_frame_for_hsv, cv.COLOR_BGR2HSV)

        # Extract the saturatin channel
        s_channel = hsv[:, :, 1]
        
        # process the saturation channel for edge detection
        blurred_s_channel = cv.GaussianBlur(s_channel, (5, 5), 0)
        _, thresholded_s_channel = cv.threshold(blurred_s_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        blurred_otsu_s_channel = cv.GaussianBlur(thresholded_s_channel, (5, 5), 0)
        canny_s_channel = cv.Canny(blurred_otsu_s_channel, 100, 200)
        s_channel_contours, _ = cv.findContours(canny_s_channel, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        
        longest_contour= identify_inner_edge(s_channel_contours,sample_second_longest_contour)

        if longest_contour is not None:
            cv.drawContours(canny, [longest_contour], -1, 255, 1)
    return original_frame,original_frame_resized,blurred_otsu,canny,blurred_image,grayscale_image



def preprocess_for_detection(image,style,sample_longest_contour,sample_second_longest_contour):
    
    display_image = image.copy()
    light_image = image.copy()
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

    #if style == "Light":
        #canny = light_gusset_fabric(light_image,canny,sample_second_longest_contour)
        
    
    # Find contours and draw the bounding box of the largest contour
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    return contours,display_image,grayscale_image,x_margins,y_margins,frame_width,frame_height,canny



def light_gusset_fabric(original_frame,canny,sample_second_longest_contour):
    hsv = cv.cvtColor(original_frame, cv.COLOR_BGR2HSV)

    # Extract the saturatin channel
    s_channel = hsv[:, :, 1]
    
    # process the saturation channel for edge detection
    blurred_s_channel = cv.GaussianBlur(s_channel, (5, 5), 0)
    _, thresholded_s_channel = cv.threshold(blurred_s_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu_s_channel = cv.GaussianBlur(thresholded_s_channel, (5, 5), 0)
    canny_s_channel = cv.Canny(blurred_otsu_s_channel, 100, 200)
    s_channel_contours, _ = cv.findContours(canny_s_channel, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    
    longest_contour= identify_inner_edge(s_channel_contours,sample_second_longest_contour)

    if longest_contour is not None:
        cv.drawContours(canny, [longest_contour], -1, 255, 1)
        cv.drawContours(canny, [sample_second_longest_contour], -1, 255, 1)
    return canny



def calculateFPS(cpu_times, end_cpu, start_cpu, last_update_time, avg_cpu_fps):
    update_interval = 1  # Update FPS every second
    cpu_time = (end_cpu - start_cpu) * 1000
    cpu_times.append(cpu_time)
    current_time = time.time()
    
    if current_time - last_update_time >= update_interval:
        avg_cpu_time = np.mean(cpu_times)
        avg_cpu_fps = 1000 / avg_cpu_time if avg_cpu_time > 0 else 0
        cpu_times.clear()  # Clear the list for the next interval
        last_update_time = current_time
    
    return avg_cpu_fps, last_update_time, cpu_times
