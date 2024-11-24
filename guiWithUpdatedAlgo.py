from customtkinter import *
import cv2 as cv
from PIL import Image, ImageTk
import serial.tools
import serial.tools.list_ports 
from mainForGUI import generateOutputFrame
import numpy as np
import time
from miscellaneous import initialize_cam,preprocess_for_detection,calculateFPS
from gussetDetection import detect_gusset
from contourID import sampleContours
from display_items import thumbnail_ganeration
import serial
from datetime import datetime

cpu_times = []
last_update_time = time.time()
avg_cpu_fps = 0  # Initialize average CPU FPS
captured = False
count = 0
 
initial_image = cv.imread("resources/loading.jpg")
sample_longest_contour = 0
sample_second_longest_contour = 0
sys_error = ""

display_live_running = False  # Flag to track the running state

# Set resolutions
display_width, display_height = 640, 360
capture_width, capture_height = 3840, 2160

# Open the webcam with low resolution using DirectShow backend
cap = initialize_cam(display_width, display_height)     

conveyor_on = False

# Lists to store gusset states and their corresponding capture times
global gusset_states
global capture_times

gusset_states = []
capture_times = []

global defected

capture_count = 0
processed_count = 0

conveyor_ready = True

last_execution_time = None
serialCom= None

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(END, message)
        self.text_widget.see(END)  # Auto-scroll to the end

    def flush(self):
        pass  # Required for compatibility with stdout

def get_available_com_ports():
    # Scan and return a list of available COM ports
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def start_serial_com():
    global serialCom
    try:
        # Try to open the serial connection
        serialCom = serial.Serial(port=str(comPort), baudrate=115200, timeout=0.1)
        
        # If successful, update button text to "Connected"
        connectButton.configure(text="Connected", fg_color="green")  # Optionally, change color to indicate success

    except serial.SerialException:
        # If the connection fails, update button text to "Failed"
        connectButton.configure(text="Disconnected", fg_color="red")  # Optionally, change color to indicate failure


def update_com_port():
    global comPort
    comPort = comPortVar.get()
    if comPort != "Select COM Port":
        connectButton.configure(state="enabled")

def process_gussets():
    global processed_count
    global conveyor_ready
    current_time = time.time()
    #print("conveyor_ready : ",conveyor_ready)

    # Check if there are gussets to process and if the processed_count is within bounds
    if len(gusset_states) == 0 or len(capture_times) == 0:
        return  # No gussets to process yet
    else:
        # If this is the first gusset, process it immediately
        if capture_count == 1 and conveyor_ready:
            if gusset_states[0]:  # Bad gusset
                bad()
            else:  # good gusset
                good()
        if processed_count < len(gusset_states):
            # Check if 20 seconds have passed for the current gusset
            elapsed_time = current_time - capture_times[processed_count]

            print("Elapsed_time : ",elapsed_time)
            #print("capture_count : ",capture_count)
            #print("processed_count : ",processed_count)

            if elapsed_time >= 20:  # 20 seconds have passed for this gusset
                conveyor_ready = True
                processed_count += 1

    # Process the next gusset if it exists
    if processed_count < len(gusset_states) and conveyor_ready:
        conveyor_ready = False
        if gusset_states[processed_count-1]:  # Good gusset
            good()
        else:  
            bad()# Bad gusset


def bad():
    serialCom.write(bytes('b', 'utf-8'))
    #print("Defective - moving backward")

def good():
    serialCom.write(bytes('g', 'utf-8'))
    #print("Non-defective - moving forward")

def toggle_conveyor_forward():
    global conveyor_on
    if conveyor_on:
        serialCom.write(bytes('s', 'utf-8'))  # Command to stop conveyor
        conveyor_forward_button.configure(text="")
        conveyor_forward_button.configure(text="Conveyor Forward")
        conveyor_start_button.configure(text="")
        conveyor_backward_button.configure(text="Start Conveyor")
        print("Conveyor stopped")
    else:
        serialCom.write(bytes('f', 'utf-8'))  # Command to start conveyor
        conveyor_forward_button.configure(text="")
        conveyor_forward_button.configure(text="Stop Conveyor")
        conveyor_start_button.configure(text="")
        conveyor_backward_button.configure(text="Stop Conveyor")
        conveyor_backward_button.configure(text="")
        conveyor_backward_button.configure(text="Stop Conveyor")
        print("Conveyor started")
    conveyor_on = not conveyor_on

def toggle_conveyor_backward():
    global conveyor_on
    if conveyor_on:
        serialCom.write(bytes('s', 'utf-8'))  # Command to stop conveyor
        conveyor_backward_button.configure(text="")
        conveyor_backward_button.configure(text="Conveyor Backward")
        print("Conveyor stopped")
    else:
        serialCom.write(bytes('r', 'utf-8'))  # Command to start conveyor
        conveyor_forward_button.configure(text="")
        conveyor_forward_button.configure(text="Stop Conveyor")
        conveyor_backward_button.configure(text="")
        conveyor_backward_button.configure(text="Stop Conveyor")
        conveyor_start_button.configure(text="")
        conveyor_backward_button.configure(text="Stop Conveyor")
        print("Conveyor started")
    conveyor_on = not conveyor_on


#define the live diplay function for displaying live eed from the camera
def displayLive():
    
    process_gussets()
    
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
    if not success or image is None:
        print("Warning: live display failed.")
        return None
    else:
        gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Count non-zero pixels
        non_zero_count = cv.countNonZero(gray_frame)
        if(non_zero_count == 0):
            print("Resetting camera feed")
            cap.set(cv.CAP_PROP_FRAME_WIDTH, display_width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, display_height)
            time.sleep(0.01)  # Allow the camera to adjust
        sys_error = "Camera ready"
    #preprocessing the low res images for gusset detection process
    #cv.imshow("live",image)
    contours, display_image, grayscale_image, x_margins, y_margins, frame_width, frame_height, canny = preprocess_for_detection(image)

    #gusset detection using the contours identified
    gussetIdentified, cx, cy, box, longest_contour, display_image, grayscale_image, captured, ma, MA,confidence = detect_gusset(contours, display_image, grayscale_image, x_margins, y_margins, frame_width, frame_height, captured, canny,sample_longest_contour,sample_second_longest_contour)
    
    

    #process handling for status of gusset identification
    if gussetIdentified:
        #check if the center of the gusset is passed the center line of the frame and if the current gusset is captured before
        if cx > (frame_width / 2) and not captured:
            #set the captured status to true and display the captured image.
            captured = True
            toggle_conveyor_forward()
            displayCaptured()
            toggle_conveyor_backward()
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
    
    
    frame = display_image.copy()
    if frame is None:
        frame = initial_image

    frame_height, frame_width,_= frame.shape


    if frame_height / frame_width < 1:
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        canny = cv.rotate(canny, cv.ROTATE_90_CLOCKWISE)
        frame_resized = cv.resize(frame, (360, 640))
        canny_resized = cv.resize(canny, (360, 640))
    else:
        frame_resized = cv.resize(frame, (640, 360))
        canny_resized = cv.resize(canny, (640, 360))

    cv.putText(frame_resized, str(display_image.shape), (10, 10), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv.LINE_AA)

    camera_frame = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(camera_frame)
    photo_image = CTkImage(light_image=captured_image, size=(cameraView.winfo_width(), cameraView.winfo_height()))
    cameraView.photo_image = photo_image
    cameraView.configure(image=photo_image)




    # Convert image from one color space to other 
    live_canny_display = cv.cvtColor(canny_resized, cv.COLOR_BGR2RGBA) 
    # Capture the latest frame and transform to image 
    live_canny_display_Img = Image.fromarray(live_canny_display) 
    live_canny_display_Img_photo_image = CTkImage(light_image=live_canny_display_Img, size=(liveCannyView.winfo_width(), liveCannyView.winfo_height()))
    # Displaying photoimage in the label 
    liveCannyView.photo_image = live_canny_display_Img_photo_image
    # Configure image in the label 
    liveCannyView.configure(image=live_canny_display_Img_photo_image) 

    fpsText.configure(text=f"FPS : {int(avg_cpu_fps)}")

    cameraView.after(10, displayLive)


def displayCaptured():

    defected = False
    global capture_count
    now = datetime.now()

    captured_time = now.strftime("%Y%m%d_%H%M%S")

    capture_times.append(time.time())

    if not display_live_running:
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, capture_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, capture_height)
    time.sleep(0.01)  # Allow the camera to adjust
    
    ret, captured_frame = cap.read()
    print("ret is ",ret)
    if not ret or captured_frame is None:
        print("Warning: Frame capture failed.")
        return None
    else:
        print(f"captured resolution is:{captured_frame.shape}")
        cap.set(cv.CAP_PROP_FRAME_WIDTH, display_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, display_height)
        print(f"Switched to:{cap.get(cv.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv.CAP_PROP_FRAME_HEIGHT)}")
        time.sleep(0.01)  # Allow the camera to adjust

    if captured_frame is None:
        captured_frame = initial_image

    frame_height, frame_width, channels = captured_frame.shape
    print("resolution = "+str(frame_height)+"x"+str(frame_width))

    if frame_height/frame_width < 1:
        captured_frame = cv.rotate(captured_frame, cv.ROTATE_90_CLOCKWISE) 


    processed_frame,balance_out,fabric_side,gusset_side,fabric_damage,blurred_otsu = generateOutputFrame(captured_frame,sample_longest_contour,sample_second_longest_contour,styleValue,adhesiveWidth,colour,captured_time)

    cv.putText(processed_frame, str(captured_frame.shape), (10, 40), cv.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3, cv.LINE_AA)
    if processed_frame is None:
        print("Error: File not found.")
    else:        
        # Set the width and height 
        processed_frame_resized = cv.resize(processed_frame, (360, 640))

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



        # Convert image from one color space to other 
        captured_adhesive = cv.cvtColor(blurred_otsu, cv.COLOR_BGR2RGBA) 
        # Capture the latest frame and transform to image 
        captured_adhesive_Img = Image.fromarray(captured_adhesive) 
        captured_adhesive_Img_photo_image = CTkImage(light_image=captured_adhesive_Img, size=(captured_adhesiveView.winfo_width(), captured_adhesiveView.winfo_height()))
        # Displaying photoimage in the label 
        captured_adhesiveView.photo_image = captured_adhesive_Img_photo_image
        # Configure image in the label 
        captured_adhesiveView.configure(image=captured_adhesive_Img_photo_image) 


        
        gussetSideText.configure(text=f"Gusset side : {gusset_side}")
        sideMixupText.configure(text=f"Fabric side : {fabric_side}")
        fabricDamageText.configure(text=f"Fabric state : {fabric_damage}")

        if gusset_side == "defective":
            defected = True
        elif gusset_side == "Front":
            balanceOutText.configure(text=f"")
            if(fabric_damage == "Damaged"):
                defected = True

            gusset_states.append(defected)  
            capture_count = capture_count+1
            # Start the timer to process gussets after 20 seconds
        elif gusset_side == "Back":
            balanceOutText.configure(text=f"Adhesive tape : {balance_out}")
            if(balance_out == "Balance out" or fabric_damage == "Damaged"):
                defected = True


            gusset_states.append(defected)  
            capture_count = capture_count+1
            # Start the timer to process gussets after 20 seconds




def displayCapturedManual(captured_frame):

    defected = False
    global capture_count
    now = datetime.now()

    captured_time = now.strftime("%Y%m%d_%H%M%S")

    capture_times.append(time.time())

    if captured_frame is None:
        captured_frame = initial_image

    frame_height, frame_width, channels = captured_frame.shape
    print("resolution = "+str(frame_height)+"x"+str(frame_width))

    if frame_height/frame_width < 1:
        captured_frame = cv.rotate(captured_frame, cv.ROTATE_90_CLOCKWISE) 


    processed_frame,balance_out,fabric_side,gusset_side,fabric_damage,blurred_otsu = generateOutputFrame(captured_frame,sample_longest_contour,sample_second_longest_contour,styleValue,adhesiveWidth,colour,captured_time)

    cv.putText(processed_frame, str(captured_frame.shape), (10, 40), cv.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3, cv.LINE_AA)
    if processed_frame is None:
        print("Error: File not found.")
    else:        
        # Set the width and height 
        processed_frame_resized = cv.resize(processed_frame, (360, 640))

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



        # Convert image from one color space to other 
        captured_adhesive = cv.cvtColor(blurred_otsu, cv.COLOR_BGR2RGBA) 
        # Capture the latest frame and transform to image 
        captured_adhesive_Img = Image.fromarray(captured_adhesive) 
        captured_adhesive_Img_photo_image = CTkImage(light_image=captured_adhesive_Img, size=(captured_adhesiveView.winfo_width(), captured_adhesiveView.winfo_height()))
        # Displaying photoimage in the label 
        captured_adhesiveView.photo_image = captured_adhesive_Img_photo_image
        # Configure image in the label 
        captured_adhesiveView.configure(image=captured_adhesive_Img_photo_image) 


        
        gussetSideText.configure(text=f"Gusset side : {gusset_side}")
        sideMixupText.configure(text=f"Fabric side : {fabric_side}")
        fabricDamageText.configure(text=f"Fabric state : {fabric_damage}")

        if gusset_side == "Front":
            balanceOutText.configure(text="")
            if(fabric_damage == "Damaged"):
                defected = True

            gusset_states.append(defected)  
            capture_count = capture_count+1
            # Start the timer to process gussets after 20 seconds
        elif gusset_side == "Back":
            balanceOutText.configure(text=f"Adhesive tape : {balance_out}")
            if(balance_out == "Balance out" or fabric_damage == "Damaged"):
                defected = True
        else:
            balanceOutText.configure(text="")


            gusset_states.append(defected)  
            capture_count = capture_count+1
            # Start the timer to process gussets after 20 seconds






def toggle_display():
    global display_live_running
    display_live_running = not display_live_running
    if display_live_running:
        displayLive()
        #displayCaptured()
        startButton.configure(text="Stop")
    else:
        startButton.configure(text="Start")

def update_thumbnail():
    global sample_longest_contour
    global sample_second_longest_contour
    global styleValue
    global adhesiveWidth
    global colour
    global sys_error
    
    styleValue = style_var.get()
    adhesiveWidth = adhesiveWidth_var.get()
    colour = colour_var.get()

    if(styleValue != "Select Style" and adhesiveWidth != "Select Adhesive Width" and colour != "Select Fabric Colour" and comPort != "Select COM Port"):
        startButton.configure(state="enabled")
        uploadButton.configure(state="enabled")
    else:
        startButton.configure(state="disabled")
        uploadButton.configure(state="disabled")

    print(f"The style is {styleValue}")
    print(f"The Adhesive Width value is {adhesiveWidth}")
    print(f"The colour is {colour}")
    
    sample_path = f"images\sample\{styleValue}.jpg"
    sample_longest_contour,sample_second_longest_contour,sample_image=sampleContours(sample_path)
    thumbnail = thumbnail_ganeration(sample_longest_contour,sample_second_longest_contour,sample_image,colour,adhesiveWidth)
    # Convert image from one color space to other 
    thumbnail = cv.cvtColor(thumbnail, cv.COLOR_BGR2RGBA) 

    # Capture the latest frame and transform to image 
    thumbnail_Img = Image.fromarray(thumbnail) 

    thumbnail_Img_photo_image = CTkImage(light_image=thumbnail_Img, size=(thumbnailViewWidth, thumbnailViewHeight))


    # Displaying photoimage in the label 
    thumbnailView.photo_image = thumbnail_Img_photo_image

    # Configure image in the label 
    thumbnailView.configure(image=thumbnail_Img_photo_image) 
    
    return sample_longest_contour,sample_second_longest_contour

def upload_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select Image", 
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )
    
    if file_path:
        # Read the selected image
        uploaded_image = cv.imread(file_path)
        
        if uploaded_image is None:
            print("Error: Could not load the image.")
            return

        print(f"Image loaded from: {file_path}")

        displayCapturedManual(uploaded_image)
        
def expand():
    if expand_button._text == ">":
        expand_button.configure(text="<")
        TroubleshootingFrame.grid(row=0, column=4, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nesw")
        expandButtonFrame.grid(row=0, column=5,rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nesw")
    else:
        expand_button.configure(text=">")
        TroubleshootingFrame.grid_forget()

# Create the main application window
app = CTk()
app.title("Gusset Inspector")
app.iconbitmap("resources/logo.ico")
app.geometry("1280*1080")
# Bind the app with Escape keyboard to quit app whenever pressed
app.bind('<Escape>', lambda e: app.quit())


# Create frames to simulate borders for image views
captureFrame = CTkFrame(app,corner_radius=10, fg_color="black")
captureFrame.grid(row=0, column=1, padx=(5, 5), pady=(5, 5),stick ="nsew")

cameraFrame = CTkFrame(app,corner_radius=10, fg_color="black")
cameraFrame.grid(row=0, column=0, padx=(5, 5), pady=(5, 5),stick ="nsew")


# Set a fixed width for settingsFrame
settingsFrame = CTkFrame(app, corner_radius=10)  # Set desired width, e.g., 300
settingsFrame.grid(row=0, column=2, padx=(5, 5), pady=(5, 5), sticky="nsew")


TroubleshootingFrame = CTkFrame(app, corner_radius=10)# Hide the TroubleshootingFrame
TroubleshootingFrame.grid_forget()

previewFrame = CTkFrame(settingsFrame, corner_radius=10,fg_color="black")
previewFrame.grid(row=3, column=0, padx=(5, 5), pady=(5, 5), sticky="nwes")


statusFrame = CTkFrame(app, corner_radius=10, fg_color="black")
statusFrame.grid(row=1, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")

defectsFrame = CTkFrame(app, corner_radius=10, fg_color="black")
defectsFrame.grid(row=1, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")


expandButtonFrame = CTkFrame(app, corner_radius=3,width=4,fg_color="black")
expandButtonFrame.grid(row=0, column=4, rowspan=2,padx=(5, 5), pady=(5, 5),sticky="nsew")

conveyorStartButtonFrame = CTkFrame(app, corner_radius=3,width=4)
conveyorStartButtonFrame.grid(row=1, column=2,padx=(5, 5), pady=(5, 5),sticky="nsew")

conveyor_start_button = CTkButton(conveyorStartButtonFrame, text="Start Conveyor", command=toggle_conveyor_forward,state="disabled")
conveyor_start_button.grid(column=0, row=0,  padx=(5, 5), pady=(5, 5), stick ="nsew")

cameraViewWidth = 360
cameraViewHeight = 640

# Load initial images
initial_image_path = "resources/sample.png"  # Replace with your image path
initial_image = Image.open(initial_image_path)
#initial_image_tk = ImageTk.PhotoImage(initial_image)
initial_image_tk = CTkImage(light_image=initial_image, size=(cameraViewWidth, cameraViewHeight))

# Set specific sizes for image views
captureView = CTkLabel(captureFrame, text="", width=cameraViewWidth, height=cameraViewHeight, image=initial_image_tk)
cameraView = CTkLabel(cameraFrame, text="", width=cameraViewWidth, height=cameraViewHeight, image=initial_image_tk)

captureView.grid(row=1, column=0,padx=5, pady=5, stick="nsew")
cameraView.grid(row=1, column=0,padx=5, pady=5, stick="nsew")


# Add Labels
liveViewLabel = CTkLabel(cameraFrame, text="Live Camera View")
liveViewLabel.grid(row=0, column=0, padx=(5, 5), pady=(5, 5))

capturedViewLabel = CTkLabel(captureFrame, text="Captured View")
capturedViewLabel.grid(row=0, column=0, padx=(5, 5), pady=(5, 5))

settingsLabel = CTkLabel(settingsFrame, text="Settings")
settingsLabel.grid(row=0, column=0, padx=(5, 5), pady=(5, 5))


troubleshootLabel = CTkLabel(TroubleshootingFrame, text="Troubleshoot")
troubleshootLabel.grid(row=0, column=0,columnspan=2, padx=(5, 5), pady=(5, 5))

thumbnailViewWidth, thumbnailViewHeight = 100,150 # Replace with your actual dimensions
# Convert the PIL Image to a CTkImage
thumbnailView = CTkLabel(previewFrame, text="", width=thumbnailViewWidth, height=thumbnailViewHeight)
thumbnailView.pack(padx=5, pady=5)

# Convert the PIL Image to a CTkImage
liveCannyView = CTkLabel(TroubleshootingFrame, text="",fg_color="black",width=180,height=360)
liveCannyView.grid(row=2, column=0, padx=(5, 5), pady=(5, 5))

# Convert the PIL Image to a CTkImage
captured_adhesiveView = CTkLabel(TroubleshootingFrame, text="",fg_color="black",width=180,height=360)
captured_adhesiveView.grid(row=2, column=1, padx=(5, 5), pady=(5, 5))




# Initialize dropdown with available COM ports
available_ports = get_available_com_ports()
available_ports.insert(0, "Select COM Port")  # Default option


comPortVar = StringVar(value="Select COM Port")
comPort_dropdown_menu = CTkOptionMenu(settingsFrame,variable=comPortVar, values=available_ports,width=200)
comPort_dropdown_menu.grid(row=1, column=0, padx=(5, 5), pady=(5, 5),stick="ew")
comPortVar.trace("w", lambda *args: update_com_port())

connectButton = CTkButton(settingsFrame, text="Connect", command=start_serial_com, width=200, state="disabled")
connectButton.grid(row=2, column=0, padx=(5, 5), pady=(5, 5),stick="ew")


# Add style selector
styleLabel = CTkLabel(settingsFrame, text="Style")
styleLabel.grid(row=4, column=0, padx=(5, 5), pady=(5, 5),stick="ew")

style_var = StringVar(value="Select Style")
style_dropdown_menu = CTkOptionMenu(settingsFrame, variable=style_var, values=["CSI70", "SB70"])
style_dropdown_menu.grid(row=5, column=0, padx=(5, 5), pady=(5, 5),stick="ew")
style_var.trace("w", lambda *args: update_thumbnail())


# Add adhesiveWidth selector
adhesiveWidthLabel = CTkLabel(settingsFrame, text="Adhesive Width")
adhesiveWidthLabel.grid(row=6, column=0, padx=(5, 5), pady=(5, 5),stick="ew")

adhesiveWidth_var = StringVar(value="Select Adhesive Width")
width_dropdown_menu = CTkOptionMenu(settingsFrame, variable=adhesiveWidth_var, values=["4mm", "6mm"])
width_dropdown_menu.grid(row=7, column=0, padx=(5, 5), pady=(5, 5),stick="ew")
adhesiveWidth_var.trace("w", lambda *args: update_thumbnail())


# Add colour selector
styleLabel = CTkLabel(settingsFrame, text="Colour")
styleLabel.grid(row=8, column=0, padx=(5, 5), pady=(5, 5),stick="ew")

colour_var = StringVar(value="Select Fabric Colour")
colour_dropdown_menu = CTkOptionMenu(settingsFrame, variable=colour_var, values=["Nero", "Skin","Bianco"])
colour_dropdown_menu.grid(row=9, column=0, padx=(5, 5), pady=(5, 5),stick="ew")
colour_var.trace("w", lambda *args: update_thumbnail())


# Create a button to open the camera in GUI app
startButton = CTkButton(settingsFrame, text="Start", command=toggle_display,state="disabled")
startButton.grid(row=10, column=0, padx=(5, 5), pady=(5, 5), sticky="nesw")


liveCannyLabel = CTkLabel(TroubleshootingFrame, text='Live Contour View')
liveCannyLabel.grid(column=0, row=1, padx=(5, 5), pady=(5, 5))

predictedLabel = CTkLabel(TroubleshootingFrame, text='Predicted components')
predictedLabel.grid(column=1, row=1, padx=(5, 5), pady=(5, 5))

label = CTkLabel(TroubleshootingFrame, text='Conveyor manual controller')
label.grid(column=0, row=5, columnspan=2, padx=(5, 5), pady=(5, 5))

btn_bad = CTkButton(TroubleshootingFrame, text='Defective',command=bad,state="disabled" )
btn_bad.grid(column=1, row=6, padx=(5, 5), pady=(5, 5))

btn_good = CTkButton(TroubleshootingFrame, text='Non-defective',command=good,state="disabled")
btn_good.grid(column=0, row=6, padx=(5, 5), pady=(5, 5))

conveyor_forward_button = CTkButton(TroubleshootingFrame, text="Conveyor Forward", command=toggle_conveyor_forward,state="disabled")
conveyor_forward_button.grid(column=1, row=7,  padx=(5, 5), pady=(5, 5))

conveyor_backward_button = CTkButton(TroubleshootingFrame, text="Conveyor Backward", command=toggle_conveyor_backward,state="disabled")
conveyor_backward_button.grid(column=0, row=7,  padx=(5, 5), pady=(5, 5))

uploadButton = CTkButton(TroubleshootingFrame, text="Upload Image", command=upload_image, state="disabled")
uploadButton.grid(row=3, column=1, padx=(5, 5), pady=(5, 5))

expand_button = CTkButton(expandButtonFrame, text='>',fg_color="grey17",text_color="grey40",command=expand )
expand_button.grid(column=0, row=0, stick="nsew")


statusLabel = CTkLabel(statusFrame, text="Program Status")
statusLabel.grid(row=0, column=0, padx=(10, 10), pady=(5, 5))

fpsText = CTkLabel(statusFrame, text="")
fpsText.grid(row=0, column=1, padx=(10, 10), pady=(5, 5))

statusLabelText = CTkLabel(statusFrame, text="")
statusLabelText.grid(row=1, column=0, padx=(10, 10), pady=(5, 5))

confidenceText = CTkLabel(statusFrame, text="")
confidenceText.grid(row=1, column=1, padx=(10, 10), pady=(5, 5))

gussetSideText = CTkLabel(defectsFrame, text="")
gussetSideText.grid(row=0, column=0, padx=(10, 10), pady=(5, 5))

balanceOutText = CTkLabel(defectsFrame, text="")
balanceOutText.grid(row=1, column=0, padx=(10, 10), pady=(5, 5))

sideMixupText = CTkLabel(defectsFrame, text="")
sideMixupText.grid(row=2, column=0, padx=(10, 10), pady=(5, 5))

fabricDamageText = CTkLabel(defectsFrame, text="")
fabricDamageText.grid(row=3, column=0, padx=(10, 10), pady=(5, 5))

# Output display (CTkTextbox for print statements)
output_text = CTkTextbox(TroubleshootingFrame)
output_text.grid(row=8, column=0,columnspan=2,padx=(10, 10), pady=(5, 5),stick="nesw")

# Redirect print to the CTkTextbox
sys.stdout = RedirectText(output_text)


if (sample_longest_contour != 0  & sample_second_longest_contour != 0):
    sample_longest_contour,sample_second_longest_contour = update_thumbnail()
else:
    sys_error = "sample contour error"
# Create an infinite loop for displaying app on screen


# Adjust the layout for proportional resizing
app.grid_rowconfigure(0, weight=0)  
app.grid_rowconfigure(1, weight=1)  
app.grid_rowconfigure(2, weight=0)  
app.grid_rowconfigure(3, weight=0) 
app.grid_rowconfigure(4, weight=0)  
app.grid_rowconfigure(5, weight=0)  

app.grid_columnconfigure(0, weight=0)  
app.grid_columnconfigure(1, weight=0)  
app.grid_columnconfigure(2, weight=1) 
app.grid_columnconfigure(3, weight=0) 
app.grid_columnconfigure(4, weight=0)  


expandButtonFrame.grid_rowconfigure(0, weight=1)
settingsFrame.grid_columnconfigure(0, weight=1)
settingsFrame.grid_rowconfigure(10, weight=1)
conveyorStartButtonFrame.grid_rowconfigure(0, weight=1)
conveyorStartButtonFrame.grid_columnconfigure(0, weight=1)

# Make frames resize proportionally
captureFrame.grid_propagate(True)
cameraFrame.grid_propagate(True)
previewFrame.grid_propagate(True)
defectsFrame.grid_propagate(True)
# Fix button sizes
connectButton.grid_propagate(True)
colour_dropdown_menu.grid_propagate(True)
width_dropdown_menu.grid_propagate(True)
style_dropdown_menu.grid_propagate(True)
startButton.grid_propagate(True)
btn_bad.grid_propagate(True)
btn_good.grid_propagate(True)
conveyor_forward_button.grid_propagate(True)
conveyor_backward_button.grid_propagate(True)
uploadButton.grid_propagate(True)

settingsFrame.grid_propagate(True)

expandButtonFrame.grid_propagate(True)
expand_button.grid_propagate(True)

# Fix button sizes
connectButton.configure(height=40)
btn_bad.configure(height=40)
btn_good.configure(height=40)
conveyor_forward_button.configure(height=40)
conveyor_backward_button.configure(height=40)
uploadButton.configure(height=40)
expand_button.configure(width=10)

# Adjust TroubleshootingFrame dimensions dynamically
TroubleshootingFrame.grid_propagate(True)

# Adjust thumbnail view inside preview frame
thumbnailView.pack(fill="both", expand=True, padx=5, pady=5)
app.mainloop()