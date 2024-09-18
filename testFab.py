import cv2 as cv
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk


# Replace 'your_image_path.jpg' with the path to your image
image_path = 'm.jpg'
# Load your image using OpenCV in grayscale
captured_frame = cv.imread(image_path)

# Compute local entropy to analyze texture
grayscale_image = cv.cvtColor(captured_frame, cv.COLOR_BGR2GRAY)
disk_radius = 20  # Adjust disk radius as needed
entr_img = entropy(grayscale_image, disk(disk_radius))

# Normalize entropy image to the range [0, 1]
entr_img = (entr_img - entr_img.min()) / (entr_img.max() - entr_img.min())
entr_img_8bit = img_as_ubyte(entr_img)

processed_frame = cv.resize(entr_img, (360, 640))    
#cv.imshow("processed_frame", processed_frame) 
entr_img_8bit = cv.resize(entr_img_8bit, (360, 640))    
#cv.imshow("entr_img_8bit", entr_img_8bit)
#cv.imshow("processed_frame", processed_frame)  
##cv.imshow("entr_img_8bit", entr_img_8bit)

    # Wait indefinitely for a key press
cv.waitKey(0)