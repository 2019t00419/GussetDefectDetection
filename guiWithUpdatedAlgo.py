from customtkinter import *
import cv2 as cv
from PIL import Image, ImageTk 
from mainForGUI import generateOutputFrame
import numpy as np
import time
from miscellaneous import initialize_cam,preprocess_for_detection,calculateFPS
from gussetDetection import detect_gusset,sampleContour

cpu_times = []
last_update_time = time.time()
avg_cpu_fps = 0  # Initialize average CPU FPS
captured = False
style =  "Light"

count = 0
 
initial_image = cv.imread("resources/loading.jpg")

display_live_running = False  # Flag to track the running state

# Set resolutions
display_width, display_height = 640, 480
capture_width, capture_height = 3840, 2160



# Open the webcam with low resolution using DirectShow backend
cap = initialize_cam(display_width, display_height)
#run sampleContour one time at the start of the program
sampleContour()


#define the live diplay function for displaying live eed from the camera
def displayLive():
    #define global variables
    global captured
    global count
    global cpu_times
    global avg_cpu_fps
    global last_update_time

    #track computation time for framerate calculation
    start_cpu = time.time()

    #Read the current frame from the low res camera instance "cap"
    success, image = cap.read()

    #Error handing for failing to load current frame
    if not success:
        print("Failed to load video")
        return None

    #preprocessing the low res images for gusset detection process
    contours, display_image, grayscale_image, x_margins, y_margins, frame_width, frame_height, canny = preprocess_for_detection(image,style)
    #gusset detection using the contours identified
    gussetIdentified, cx, cy, box, longest_contour, second_longest_contour, display_image, grayscale_image, captured, ma, MA,confidence = detect_gusset(contours, display_image, grayscale_image, x_margins, y_margins, frame_width, frame_height, captured, canny)

    #process handling for status of gusset identification
    if gussetIdentified:
        #check if the center of the gusset is passed the center line of the frame and if the current gusset is captured before
        if cx > (frame_width / 2) and not captured:
            #set the captured status to true and display the captured image.
            captured = True
            displayCaptured()
            count += 1
        #update the status label and the confidence of the gusset identification
        statusLabelText.configure(text=f"Gusset detected")
        confidenceText.configure(text=f"{int(confidence)}% Confidence")
    else :
        #update the status label as searching
        statusLabelText.configure(text=f"Searching")
        confidenceText.configure(text=f"")
    #end the time tracking    
    end_cpu = time.time()
    
    #calcuate the average fps during 1 seceond
    avg_cpu_fps, last_update_time, cpu_times = calculateFPS(cpu_times, end_cpu, start_cpu, last_update_time, avg_cpu_fps)

    #display the center line on the live feed
    cv.line(display_image, (int(frame_width / 2), 0), (int(frame_width / 2), int(frame_height)), (0, 255, 0), 2)

    #modifying the resultant frame for displaying on live feed
    #display_image was replaced using canny for experimental purpose.
    frame = canny.copy()
    if frame is None:
        frame = initial_image

    frame_height, frame_width= frame.shape
    if frame_height / frame_width < 1:
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame_resized = cv.resize(frame, (480, 640))
    else:
        frame_resized = cv.resize(frame, (640, 480))

    cv.putText(frame_resized, str(display_image.shape), (10, 10), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv.LINE_AA)

    camera_frame = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(camera_frame)
    photo_image = CTkImage(light_image=captured_image, size=(cameraView.winfo_width(), cameraView.winfo_height()))
    cameraView.photo_image = photo_image
    cameraView.configure(image=photo_image)

    fpsText.configure(text=f"FPS : {int(avg_cpu_fps)}")

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


    processed_frame,balance_out,fabric_side,gusset_side = generateOutputFrame(captured_frame,style)


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

        
        gussetSideText.configure(text=f"Gusset side : {gusset_side}")
        sideMixupText.configure(text=f"Fabric side : {fabric_side}")
        if gusset_side == "Front":
            balanceOutText.configure(text=f"")
        elif gusset_side == "Back":
            balanceOutText.configure(text=f"Adhesive tape : {balance_out}")



def toggle_display():
    global display_live_running
    display_live_running = not display_live_running
    if display_live_running:
        displayLive()
        displayCaptured()
        startButton.configure(text="Stop")
    else:
        startButton.configure(text="Start")



# Create the main application window
app = CTk()

# Bind the app with Escape keyboard to quit app whenever pressed
app.bind('<Escape>', lambda e: app.quit())

# Create frames to simulate borders for image views
captureFrame = CTkFrame(app, width=490, height=650, corner_radius=10, fg_color="black")
cameraFrame = CTkFrame(app, width=490, height=650, corner_radius=10, fg_color="black")

cameraViewWidth = 480
cameraViewHeight = 640

# Load initial images
initial_image_path = "resources/sample.png"  # Replace with your image path
initial_image = Image.open(initial_image_path)
#initial_image_tk = ImageTk.PhotoImage(initial_image)
initial_image_tk = CTkImage(light_image=initial_image, size=(cameraViewWidth, cameraViewHeight))

# Set specific sizes for image views
captureView = CTkLabel(captureFrame, text="", width=cameraViewWidth, height=cameraViewHeight, image=initial_image_tk)
cameraView = CTkLabel(cameraFrame, text="", width=cameraViewWidth, height=cameraViewHeight, image=initial_image_tk)

captureView.pack(padx=5, pady=5)
cameraView.pack(padx=5, pady=5)

# Apply rowspan to both image views
captureFrame.grid(row=1, column=1, rowspan=6, padx=(10, 5), pady=(10, 5))
cameraFrame.grid(row=1, column=0, rowspan=6, padx=(0, 10), pady=(10, 5))

# Create a button to open the camera in GUI app
startButton = CTkButton(app, text="Start", command=toggle_display)
startButton.grid(row=3, column=3, padx=(10, 5), pady=(10, 5))

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
statusFrame.grid(row=4, column=2, columnspan=2, padx=(10, 5), pady=(10, 5), sticky="nsew")

defectsFrame = CTkFrame(app, corner_radius=10, fg_color="black")
defectsFrame.grid(row=5, column=2, columnspan=2, rowspan=5, padx=(10, 5), pady=(10, 5), sticky="nsew")

# Ensure the rows and columns expand proportionally
app.grid_rowconfigure(4, weight=1)
app.grid_columnconfigure(2, weight=1)
app.grid_rowconfigure(5, weight=1)
app.grid_columnconfigure(3, weight=1)

statusLabel = CTkLabel(statusFrame, text="Program Status")
statusLabel.grid(row=0, column=0, padx=(10, 10), pady=(10, 5))

fpsText = CTkLabel(statusFrame, text="")
fpsText.grid(row=0, column=1, padx=(10, 10), pady=(10, 5))

statusLabelText = CTkLabel(statusFrame, text="")
statusLabelText.grid(row=1, column=0, padx=(10, 10), pady=(10, 5))

confidenceText = CTkLabel(statusFrame, text="")
confidenceText.grid(row=1, column=1, padx=(10, 10), pady=(10, 5))

gussetSideText = CTkLabel(defectsFrame, text="")
gussetSideText.grid(row=0, column=0, padx=(10, 10), pady=(10, 5))

balanceOutText = CTkLabel(defectsFrame, text="")
balanceOutText.grid(row=1, column=0, padx=(10, 10), pady=(10, 5))

sideMixupText = CTkLabel(defectsFrame, text="")
sideMixupText.grid(row=2, column=0, padx=(10, 10), pady=(10, 5))

# Create an infinite loop for displaying app on screen
app.mainloop()