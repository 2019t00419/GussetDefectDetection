from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk 
from customtkinter import *
from mainForGUI import main




def open_camera(): 
    # Capture the video frame by frame 
    _, frame = vid.read() 
    
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
    cameraView.after(10, open_camera) 
  
def capture():
    _, captured_frame = vid.read() 

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
button1 = CTkButton(app, text="Capture", command=capture) 

button1.grid(row=1, column=1, padx=(0, 10))
open_camera()
capture()  
# Create an infinite loop for displaying app on screen 
app.mainloop() 