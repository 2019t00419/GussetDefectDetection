import os
import cv2 as cv
import numpy as np
import time
from contourID import identify_inner_edge
from detectionAssist import detection_support
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

def preprocess(original_frame, sample_longest_contour, sample_second_longest_contour, styleValue, thickness, colour):
    start_time = time.time()  # Start timer

    captured_view_image = original_frame.copy()
    threshold1 = 100
    threshold2 = 200
    
    # Detecting gusset side using assisted RF model
    #side, _, _ = detect_gusset_side(original_frame, 240, model, label_encoder)
    #print(f"Side predicted by the gusset side detection assist: {side[0]}")

    end_time0 = time.time()  # End timer
    print(f"Time taken to complete detect_gusset_side function: {end_time0 - start_time:.6f} seconds")
    # Removing background based on color and side conditions

    assisted_adhesive_image, assisted_fabric_image = detection_support(original_frame,colour)

    grayscale_image_fabric = cv.cvtColor(assisted_fabric_image,cv.COLOR_BGR2GRAY)

    #newly added line for detection support
    
    resized_image1 = cv.resize(assisted_adhesive_image, (360, 640))
    #cv.imshow("assisted adhesive image",resized_image1)

    resized_image2 = cv.resize(grayscale_image_fabric, (360, 640))
    #cv.imshow("assisted image grayscale_image_fabric",resized_image2)


    # Processing grayscale image
    blurred_assisted_adhesive_image = cv.GaussianBlur(assisted_adhesive_image, (5, 5), 0)
    canny = cv.Canny(blurred_assisted_adhesive_image, 100, 200)

    return captured_view_image,blurred_assisted_adhesive_image,canny


def preprocess_for_detection(image):

    display_image = image.copy()
    #grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    """
    if(colour == "Bianco" or colour == "Skin"):
        grayscale_image = image[:, :, 2] #Red channel
        detection_mask = np.zeros_like(grayscale_image)
        #grayscale_image = image[:, :, 2] #Blue channel
        #grayscale_image = hsv_image[:, :, 1]
    elif(colour == "Nero"):
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        grayscale_image = hsv[:, :, 1]
        detection_mask = np.ones_like(grayscale_image)
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    grayscale_image = hsv[:, :, 1]
    detection_mask = np.ones_like(grayscale_image)
    ##cv.imshow("grayscale_image",grayscale_image)

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

    #cv.imshow("masked_grayscale_image",masked_grayscale_image)
    # Show the masked grayscale image
    # CPU operations
    blurred_image = cv.GaussianBlur(masked_grayscale_image, (5, 5), 0)
    _, cpu_thresholded_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(cpu_thresholded_image, (5, 5), 0)
    
    canny = cv.Canny(blurred_otsu, 100, 200)
    #cv.imshow("canny",canny)
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
