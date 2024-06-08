from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk 
from customtkinter import *
from mainForGUI import main
from ultralytics import YOLO
import numpy as np
import time

cap = cv.VideoCapture(0)  # Use the webcam

cpu_times = []
last_update_time = time.time()
update_interval = 1  # Update FPS every second
avg_cpu_fps = 0  # Initialize average CPU FPS
captured = False
detection_length = 600
detection_height = 460

count = 0
 

initial_image = cv.imread("resources\loading.jpg")

def displayLive():
    
    global captured
    global count
    global cpu_times
    global avg_cpu_fps
    global last_update_time

    frame_start_time = time.time()
    
    start_cpu = time.time()
    start_open = time.time()
    success, image = cap.read()
    display_image = image.copy() if success else None
    if not success:
        print("Failed to load video")
        return None

    end_open = time.time()
    open_time = (end_open - start_open) * 1000
    print("Open time : " + str(open_time) + "ms")
    
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    detection_mask = np.zeros_like(grayscale_image)

    frame_height, frame_width = grayscale_image.shape
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
    
    # Find contours and draw the bounding box of the largest contour
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)

        x, y, w, h = cv.boundingRect(largest_contour)
        cv.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if (x < x_margins) or (y < y_margins) or ((x+w) > (frame_width-x_margins)) or ((y+h) > (frame_height-y_margins)):
            captured = False
        else:
            rect = cv.minAreaRect(largest_contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            M = cv.moments(box)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv.circle(display_image, (cx, cy), 5, (255, 0, 0), cv.FILLED)
            if cx > (frame_width / 2) and not captured:
                captured = True
                displayCaptured()
                count += 1

            cv.drawContours(display_image, [box], 0, (0, 0, 255), 2)

            (x, y), (MA, ma), angle = cv.fitEllipse(largest_contour)
            cv.putText(display_image, f"Major axis length: {int(MA)}    Minor axis length: {int(ma)}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            
            # Create a mask for the largest contour
            mask = np.zeros_like(grayscale_image)
            cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)
            cv.drawContours(canny, [largest_contour], -1, 0, 2)

            # Mask the Canny image
            masked_canny = cv.bitwise_and(canny, canny, mask=mask)

            # Find contours inside the masked canny image
            inner_contours, _ = cv.findContours(masked_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if inner_contours:
                largest_inner_contour = max(inner_contours, key=cv.contourArea)
                ret = cv.matchShapes(largest_inner_contour,largest_contour,1,0.0)
                if ret<0.5:
                    cv.drawContours(display_image, [largest_inner_contour], -1, (255, 0, 255), 1)
                    cv.drawContours(display_image, [largest_contour], -1, (255, 0, 0), 1)

    # Update average FPS every second
    end_cpu = time.time()
    cpu_time = (end_cpu - start_cpu) * 1000
    cpu_times.append(cpu_time)
    print("CPU time : " + str(cpu_time) + "ms")
    current_time = time.time()
    if current_time - last_update_time >= update_interval:
        avg_cpu_time = np.mean(cpu_times)
        avg_cpu_fps = 1000 / avg_cpu_time if avg_cpu_time > 0 else 0

        print("Average CPU FPS : " + str(avg_cpu_fps))
        cpu_times = []
        last_update_time = current_time

    # Display original image with bounding box, average FPS, and average color
    cv.putText(display_image, f"CPU FPS: {avg_cpu_fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.line(display_image, (int(frame_width/2), 0), (int(frame_width/2), int(frame_height)), (0, 255, 0), 2)


    frame = display_image.copy()

    if frame is None:
        frame = initial_image

    frame_height, frame_width, channels = frame.shape
    if frame_height/frame_width < 1:
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame_resized = cv.resize(frame, (480, 640))
    else:
        frame_resized = cv.resize(frame, (640, 480))
  
    # Convert image from one color space to other 
    camera_frame = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGBA)
  
    # Capture the latest frame and transform to image 
    captured_image = Image.fromarray(camera_frame) 
  
    # Convert captured image to photoimage 
    photo_image = ImageTk.PhotoImage(image=captured_image) 
  
    # Displaying photoimage in the label 
    cameraView.photo_image = photo_image 
  
    # Configure image in the label 
    cameraView.configure(image=photo_image) 
  
    # Repeat the same process after every 10 seconds 
    cameraView.after(10, displayLive) 


  
def displayCaptured():

    ret, captured_frame = cap.read()
    if ret:
        cv.imwrite(f"Images/captured/captured ({count}).jpg", captured_frame)
    else:
        print("Failed to capture image")

    if captured_frame is None:
        captured_frame = initial_image

    frame_height, frame_width, channels = captured_frame.shape
    print("resolution = "+str(frame_height)+"x"+str(frame_width))

    if frame_height/frame_width < 1:
        captured_frame = cv.rotate(captured_frame, cv.ROTATE_90_CLOCKWISE) 
    cv.imwrite("images\in\captured\Captured ("+str(0)+").jpg",captured_frame)

    processed_frame = main(captured_frame)
    if processed_frame is None:
        print("Error: File not found.")
    else:     
                
        # Set the width and height 
        processed_frame_resized = cv.resize(processed_frame, (480, 640))

        # Convert image from one color space to other 
        processed_frame_resized = cv.cvtColor(processed_frame_resized, cv.COLOR_BGR2RGBA) 
    
        # Capture the latest frame and transform to image 
        processed_frame_resized_image = Image.fromarray(processed_frame_resized) 
    
        # Convert captured image to photoimage 
        processed_photo_image = ImageTk.PhotoImage(image=processed_frame_resized_image) 
    
        # Displaying photoimage in the label 
        captureView.photo_image = processed_photo_image
    
        # Configure image in the label 
        captureView.configure(image=processed_photo_image) 

# Define a video capture object 
vid = cv.VideoCapture(0) 
#vid = cv.VideoCapture("images\in\sample_long.mp4")  
# Declare the width and height in variables 
width, height = 3024, 4032
  
# Set the width and height 
vid.set(cv.CAP_PROP_FRAME_WIDTH, width) 
vid.set(cv.CAP_PROP_FRAME_HEIGHT, height) 


# Create a GUI app 
app = CTk() 
# Bind the app with Escape keyboard to 
# quit app whenever pressed 
app.bind('<Escape>', lambda e: app.quit()) 
  
# Create a label and display it on app 

captureView = Label(app) 
cameraView = Label(app) 

captureView.grid(row=0, column=0, padx=(10, 5))
cameraView.grid(row=0, column=1, padx=(0, 10))

# Create a function to open camera and 
# display it in the label_widget on app 
  
# Create a button to open the camera in GUI app 
button1 = CTkButton(app, text="Capture", command=displayCaptured) 

button1.grid(row=1, column=1, padx=(0, 10))
displayLive()
displayCaptured() 
# Create an infinite loop for displaying app on screen 
app.mainloop() 