import os
import cv2 as cv
import numpy as np
import time
from contourID import identify_inner_edge
from detectionAssist import detection_support
from contourID import identify_outer_edge
from datetime import datetime
from gussetSideDetectionAssist import detect_gusset_side
import pickle
from errorHandler import show_error

# Load the trained model and label encoder (no need to load each time for multiple predictions)
model = pickle.load(open("detectionSupportModelforSide", 'rb'))
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))


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


def preprocess(original_frame,sample_longest_contour,sample_second_longest_contour,styleValue,thickness,colour):
    
    threshold1=100
    threshold2=200
#removing background
    if(colour == "Bianco" or colour == "Skin"):
        grayscale_image = original_frame[:, :, 2] #Red channel
        #grayscale_image = image[:, :, 2] #Blue channel
        #grayscale_image = hsv_image[:, :, 1]
    elif(colour == "Nero"):
        grayscale_image = original_frame[:, :, 1] #Green channel

    # Show the masked grayscale image
    # CPU operations
    blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)
    _, thresholded_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(thresholded_image, (5, 5), 0)
    canny = cv.Canny(blurred_otsu, 100, 200)

    cv.imshow("canny",canny)
    # Find contours and draw the bounding box of the largest contour
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    longest_contour=identify_outer_edge(contours,sample_longest_contour)

    if longest_contour is not None:
        gusset_boundary_mask_colour = np.zeros_like(original_frame)
        cv.drawContours(gusset_boundary_mask_colour, [longest_contour], -1, (255,255,255), cv.FILLED)

        gusset_boundary_mask = cv.cvtColor(gusset_boundary_mask_colour, cv.COLOR_BGR2GRAY)

        
        original_frame_background_removed = cv.bitwise_and(original_frame, gusset_boundary_mask_colour, mask=gusset_boundary_mask)

        cv.imshow("original_frame_background_removed",original_frame_background_removed)

        # Get the current date and time
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

        # Define the filename with timestamp
        filename = f"images/captured/captured_{timestamp}.jpg"
        # Save the image
        cv.imwrite(filename, original_frame_background_removed)

        
        prediction, _, _ = detect_gusset_side(original_frame_background_removed, 240, model, label_encoder)
        print(f"Side predicted by the gusset side detection assist : {prediction[0]}")


    #None assisted detection result
        original_grayscale_image = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)
        # Apply Gaussian Blur
        original_blurred_image = cv.GaussianBlur(original_grayscale_image, (5, 5), 0)
        # Otsu's Binarization
        _, original_otsu_thresholded = cv.threshold(original_blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        original_blurred_otsu = cv.GaussianBlur(original_otsu_thresholded, (5, 5), 0)


        if(prediction[0] == 'back'):
        #detection assisted image from machine learning
            assisted_grayscale_image = detection_support(original_frame)
            #original_frame_resized = cv.resize(original_frame, (720, 1280))
            # Apply Gaussian Blur
            assisted_blurred_image = cv.GaussianBlur(assisted_grayscale_image, (5, 5), 0)
            # Otsu's Binarization
            _, assisted_otsu_thresholded = cv.threshold(assisted_blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            assisted_blurred_otsu = cv.GaussianBlur(assisted_otsu_thresholded, (5, 5), 0)

            
        #combining both images to increase accuracy of the selection area
            blurred_otsu = cv.bitwise_and(assisted_blurred_otsu, original_blurred_otsu, mask=original_blurred_otsu)


        # Apply Canny edge detection
            canny = cv.Canny(blurred_otsu, threshold1, threshold2)
        elif (prediction == 'front'):
        
        # Apply Canny edge detection
            canny = cv.Canny(original_blurred_otsu, threshold1, threshold2)
        #cv.imshow('Canny Edge', canny)
        
        if (colour == "Bianco" or colour == "Skin"):
            #original_frame_for_hsv = cv.resize(original_frame, (720, 1280))
            #hsv = cv.cvtColor(original_frame_for_hsv, cv.COLOR_BGR2HSV)
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
                cv.imshow("canny+NEW",canny)
        else:  
            show_error("No outer edge identified")
    return original_frame,blurred_otsu,canny



def preprocess_for_detection(image,sample_longest_contour,sample_second_longest_contour,styleValue,thickness,colour):

    display_image = image.copy()
    #grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if(colour == "Bianco" or colour == "Skin"):
        grayscale_image = image[:, :, 2] #Red channel
        #grayscale_image = image[:, :, 2] #Blue channel
        #grayscale_image = hsv_image[:, :, 1]
    elif(colour == "Nero"):
        grayscale_image = image[:, :, 1] #Green channel

    #cv.imshow("grayscale_image",grayscale_image)
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
