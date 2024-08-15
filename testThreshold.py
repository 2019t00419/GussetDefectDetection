from customtkinter import *
import cv2 as cv
from mainForGUI import generateOutputFrame
import time
from contourID import sampleContours
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk


cpu_times = []
last_update_time = time.time()
avg_cpu_fps = 0  # Initialize average CPU FPS
captured = False
style =  "Light" #Light or Dark

count = 0
 
sample_path = "images\sample\sample (0).jpg"
initial_image = cv.imread("resources/loading.jpg")

display_live_running = False  # Flag to track the running state

# Replace 'your_image_path.jpg' with the path to your image
image_path = 'h.jpg'
# Load your image using OpenCV in grayscale
captured_frame = cv.imread(image_path)

sample_longest_contour,sample_second_longest_contour=sampleContours(sample_path)

if captured_frame is None:
    captured_frame = initial_image

frame_height, frame_width, channels = captured_frame.shape
print("resolution = "+str(frame_height)+"x"+str(frame_width))

if frame_height/frame_width < 1:
    captured_frame = cv.rotate(captured_frame, cv.ROTATE_90_CLOCKWISE) 

# Compute local entropy to analyze texture
grayscale_image = cv.cvtColor(captured_frame, cv.COLOR_BGR2GRAY)
disk_radius = 20  # Adjust disk radius as needed
entr_img = entropy(grayscale_image, disk(disk_radius))

# Normalize entropy image to the range [0, 1]
entr_img = (entr_img - entr_img.min()) / (entr_img.max() - entr_img.min())
entr_img_8bit = img_as_ubyte(entr_img)

processed_frame,balance_out,fabric_side,gusset_side = generateOutputFrame(captured_frame,style,sample_longest_contour,sample_second_longest_contour)

processed_frame = cv.resize(processed_frame, (360, 640))    
cv.imshow("processed_frame", processed_frame) 
entr_img_8bit = cv.resize(entr_img_8bit, (360, 640))    
cv.imshow("entr_img_8bit", entr_img_8bit)

    # Wait indefinitely for a key press
cv.waitKey(0)

    # Optionally, destroy all windows after the key press
cv.destroyAllWindows()
