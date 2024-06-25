from customtkinter import *
import cv2 as cv
from PIL import Image, ImageTk 
from mainForGUI import generateOutputFrame
import numpy as np
import time
from miscellaneous import initialize_cam,preprocess_for_detection,calculateFPS
from gussetDetection import detect_gusset,rem

cpu_times = []
last_update_time = time.time()
avg_cpu_fps = 0  # Initialize average CPU FPS
captured = False

count = 0
 
initial_image = cv.imread("resources/loading.jpg")

display_live_running = False  # Flag to track the running state

# Set resolutions
display_width, display_height = 640, 480
capture_width, capture_height = 3840, 2160



# Open the webcam with low resolution using DirectShow backend
cap = initialize_cam(display_width, display_height)


def displayLive():

    #Actions initialization and setting
    MA, ma = None,None
    if not display_live_running:
        return
    
    global captured
    global count
    global cpu_times
    global avg_cpu_fps
    global last_update_time
    
    start_cpu = time.time()
    start_open = time.time()
    success, image = cap.read()
    if not success:
        print("Failed to load video")
        return None

    end_open = time.time()
    open_time = (end_open - start_open) * 1000
    #print("Open time : " + str(open_time) + "ms")
    #print(image.shape)
    contours,display_image,grayscale_image,x_margins,y_margins,frame_width,frame_height,canny = preprocess_for_detection(image)
    gussetIdentified,cx,cy,box,longest_contour,second_longest_contour,display_image,grayscale_image,captured = detect_gusset(contours,display_image,grayscale_image,x_margins,y_margins,frame_width,frame_height,captured)
    if gussetIdentified :
        if cx > (frame_width / 2) and not captured:
            captured = True
            displayCaptured()
            count += 1
        ma,MA=rem(cx,cy,box,longest_contour,second_longest_contour,display_image,grayscale_image,canny)
    
            
    #print(captured)
    # Update average FPS every second
    end_cpu = time.time()

    avg_cpu_fps = calculateFPS(cpu_times,end_cpu,start_cpu,last_update_time)
    

    # Display original image with bounding box, average FPS, and average color
    #cv.putText(display_image, f"CPU FPS: {avg_cpu_fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
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
    
    cv.putText(frame_resized, str(display_image.shape), (10, 10), cv.FONT_HERSHEY_PLAIN, 0.8, (255,255,255), 1, cv.LINE_AA)
            
    
    # Convert image from one color space to other 
    camera_frame = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGBA)
  
    # Capture the latest frame and transform to image 
    captured_image = Image.fromarray(camera_frame) 
  
    # Convert captured image to photoimage 
    #photo_image = ImageTk.PhotoImage(image=captured_image) 

    # Convert captured image to CTkImage
    photo_image = CTkImage(light_image=captured_image, size=(cameraView.winfo_width(), cameraView.winfo_height()))

  
    # Displaying photoimage in the label 
    cameraView.photo_image = photo_image 
  
    # Configure image in the label 
    cameraView.configure(image=photo_image) 

    if ma is not None:
        statusLabelText.configure(text=f"FPS : {int(avg_cpu_fps)} \n \nGusset detected\nMajor axis length: {int(ma)}    Minor axis length: {int(MA)}")
    else  :
        statusLabelText.configure(text=f"FPS : {int(avg_cpu_fps)}")
            
  
    # Repeat the same process after every 10 seconds 
    cameraView.after(10, displayLive) 




def displayCaptured():
    
    ThicknessValue = dropdown_var.get()
    print(f"The thickness value is {ThicknessValue}")
    if not display_live_running:
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, capture_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, capture_height)
    time.sleep(0.01)  # Allow the camera to adjust
    
    ret, captured_frame = cap.read()
    if ret:
        print(f"captured resolution is:{captured_frame.shape}")
        cv.imwrite(f"Images/captured/captured ({count}).jpg", captured_frame)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, display_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, display_height)
        time.sleep(0.01)  # Allow the camera to adjust
    else:
        print("Failed to capture image")

    if captured_frame is None:
        captured_frame = initial_image

    frame_height, frame_width, channels = captured_frame.shape
    print("resolution = "+str(frame_height)+"x"+str(frame_width))

    if frame_height/frame_width < 1:
        captured_frame = cv.rotate(captured_frame, cv.ROTATE_90_CLOCKWISE) 
    cv.imwrite("images/in/captured/Captured ("+str(0)+").jpg",captured_frame)


    processed_frame,balance_out,fabric_side = generateOutputFrame(captured_frame)


    cv.putText(processed_frame, str(captured_frame.shape), (10, 20), cv.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2, cv.LINE_AA)
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
        #processed_photo_image = ImageTk.PhotoImage(image=processed_frame_resized_image) 
        
        processed_photo_image = CTkImage(light_image=processed_frame_resized_image, size=(cameraView.winfo_width(), cameraView.winfo_height()))

    
        # Displaying photoimage in the label 
        captureView.photo_image = processed_photo_image
    
        # Configure image in the label 
        captureView.configure(image=processed_photo_image) 

        if balance_out:
            balanceOutText.configure(text=f"Adhesive tape : Balance out")
        else  :
            balanceOutText.configure(text=f"Adhesive tape : No Issues")

        sideMixupText.configure(text=f"Fabric side : {fabric_side}")



def toggle_display():
    global display_live_running
    display_live_running = not display_live_running
    if display_live_running:
        displayLive()
        displayCaptured()
        startButton.configure(text="Stop")
    else:
        startButton.configure(text="Start")


# Create a GUI app
app = CTk()
# Bind the app with Escape keyboard to
# quit app whenever pressed
app.bind('<Escape>', lambda e: app.quit())

# Create frames to simulate borders for image views
captureFrame = CTkFrame(app, width=490, height=650, corner_radius=10, fg_color="black")
cameraFrame = CTkFrame(app, width=490, height=650, corner_radius=10, fg_color="black")

# Set specific sizes for image views
captureView = CTkLabel(captureFrame,text="", width=480, height=640)
cameraView = CTkLabel(cameraFrame,text="", width=480, height=640)

captureView.pack(padx=5, pady=5)
cameraView.pack(padx=5, pady=5)

# Apply rowspan to both image views
captureFrame.grid(row=1, column=1, rowspan=6, padx=(10, 5), pady=(10, 5))
cameraFrame.grid(row=1, column=0, rowspan=6, padx=(0, 10), pady=(10, 5))

# Create a button to open the camera in GUI app
startButton = CTkButton(app, text="Start", command=toggle_display)
startButton.grid(row=3, column=3,padx=(10, 5), pady=(10, 5))

# Add Labels
liveViewLabel = CTkLabel(app, text="Live Camera View")
liveViewLabel.grid(row=0, column=0, padx=(10, 5), pady=(10, 5))

capturedViewLabel = CTkLabel(app, text="Captured View")
capturedViewLabel.grid(row=0, column=1, padx=(10, 5), pady=(10, 5))


settingsLabel = CTkLabel(app, text="Settings")
settingsLabel.grid(row=0, column=2, columnspan=2, padx=(10, 5), pady=(10, 5))


thicknessLabel = CTkLabel(app, text="Thickness")
thicknessLabel.grid(row=1, column=2, padx=(10, 5), pady=(10, 5))

styleLabel = CTkLabel(app, text="Style")
styleLabel.grid(row=2, column=2, padx=(10, 5), pady=(10, 5))
styleentry1 = CTkEntry(app)
styleentry1.grid(row=2, column=3, padx=(10, 5), pady=(10, 5))

# Add dropdown menu
dropdown_var = StringVar(value="4mm")
dropdown_menu = CTkOptionMenu(app, variable=dropdown_var, values=["4mm", "6mm"])
dropdown_menu.grid(row=1, column=3, padx=(10, 5), pady=(10, 5))


statusFrame = CTkFrame(app, corner_radius=10, fg_color="black")
statusFrame.grid(row=4, column=2, columnspan=2, rowspan=3, padx=(10, 5), pady=(10, 5), sticky="nsew")
# Ensure the rows and columns expand proportionally
app.grid_rowconfigure(4, weight=1)
app.grid_columnconfigure(2, weight=1)
app.grid_rowconfigure(5, weight=1)
app.grid_columnconfigure(3, weight=1)

statusLabel = CTkLabel(statusFrame, text="Status")
statusLabel.grid(row=0, column=0, padx=(10, 10), pady=(10, 5) )

statusLabelText = CTkLabel(statusFrame, text="Status will appear here")
statusLabelText.grid(row=1, column=0, padx=(10, 10), pady=(10, 5) )

balanceOutText = CTkLabel(statusFrame, text="Balance out detection status will appear here")
balanceOutText.grid(row=2, column=0, padx=(10, 10), pady=(10, 5) )

sideMixupText = CTkLabel(statusFrame, text="Side mixup detection status will appear here")
sideMixupText.grid(row=3, column=0, padx=(10, 10), pady=(10, 5) )

#actions

# Create an infinite loop for displaying app on screen
app.mainloop()