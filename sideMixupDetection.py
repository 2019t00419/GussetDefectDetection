import numpy as np
import cv2 as cv
import pickle
import time
from sideMixupDetectionFilters import resizer, feature_extractor
from datetime import datetime

# Load the trained model and label encoder (no need to load each time for multiple predictions)
model = pickle.load(open("sideMixupdetectionModel", 'rb'))
label_encoder = pickle.load(open("SMD_encoder.pkl", 'rb'))
image_width = 100

def side_mixup_detection(imgc, size, model, label_encoder):
    """
    Predict the class of a single image with optimized preprocessing.
    """
    # Start time for preprocessing
    start_preprocess = time.time()
    
    # Efficient image loading and preprocessing using OpenCV
    #imgc = cv.imread(image_path, cv.IMREAD_COLOR)  # Read the image
    imgc = resizer(imgc, size)  # Resize the image

    # Convert image to grayscale and normalize
    img = cv.cvtColor(imgc, cv.COLOR_BGR2GRAY)
    img = img / 255.0  # Normalize pixel values to [0,1]
    
    # Expand dimensions and extract features
    input_img = np.expand_dims(img, axis=0)  # Prepare for feature extraction
    input_img_features = feature_extractor(input_img, imgc)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    
    # Reshape features for the model
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
    
    # End time for preprocessing
    end_preprocess = time.time()
    preprocess_time = end_preprocess - start_preprocess

    # Start time for prediction
    start_predict = time.time()
    
    # Make the prediction
    img_prediction = model.predict(input_img_for_RF)

    # Reverse label encoding to get the class name
    img_prediction = label_encoder.inverse_transform(img_prediction.ravel())  # Ensure this is 1D

    
    # End time for prediction
    end_predict = time.time()
    predict_time = end_predict - start_predict
    
    return img_prediction, preprocess_time, predict_time
"""
# Example prediction for a single image
image_path = "test/SMD/Test_images/New_0001.jpg"
prediction, preprocess_time, predict_time = side_mixup_detection(image_path, 64, model, label_encoder)

# Print the results
print(f"The predicted class for the image is: {prediction[0]}")
print(f"Preprocessing time: {preprocess_time:.6f} seconds")
print(f"Prediction time: {predict_time:.6f} seconds")
"""
def crop_image(original_frame, longest_contour, image_width):
    # Check if contour is valid
    if longest_contour is None or len(longest_contour) == 0:
        print("Warning: No valid contour provided to crop_image.")
        return None

    # Calculate moments for the center
    M = cv.moments(longest_contour)
    if M['m00'] == 0:
        print("Warning: Contour moments calculation resulted in m00 = 0.")
        return None
    
    # Get dimensions of original frame
    frame_height, frame_width, _ = original_frame.shape
    print(f"Original frame dimensions: width={frame_width}, height={frame_height}")

    # Calculate the center point (cx, cy) based on moments
    cx = int(frame_width * (M['m10'] / M['m00']) / 960)
    cy = int(frame_height * (M['m01'] / M['m00']) / 1280)
    print(f"Center point of crop: ({cx}, {cy})")

    # Define the top-left and bottom-right coordinates of the square crop
    tlx, tly = cx - int(image_width / 2), cy - int(image_width / 2)
    brx, bry = cx + int(image_width / 2), cy + int(image_width / 2)
    print(f"Initial bounding box - Top Left: ({tlx}, {tly}), Bottom Right: ({brx}, {bry})")

    # Ensure bounding box is within frame boundaries
    tlx, tly = max(0, tlx), max(0, tly)
    brx, bry = min(frame_width, brx), min(frame_height, bry)
    print(f"Adjusted bounding box - Top Left: ({tlx}, {tly}), Bottom Right: ({brx}, {bry})")

    # Check if adjusted box dimensions still form a valid area
    if tlx >= brx or tly >= bry:
        print("Warning: Adjusted crop coordinates result in an empty area. Skipping this frame.")
        return None

    # Crop the image
    cropped_image = original_frame[tly:bry, tlx:brx]
    if cropped_image.size == 0:
        print("Warning: Cropped image is empty after applying ROI.")
        return None

    # Convert cropped image to grayscale
    grayscale_cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    # Save cropped image for debugging purposes
    cv.imwrite(f"images/out/cropped/cropped ({timestamp}).jpg", grayscale_cropped_image)

    # Call fabric side detection and log timing information
    fabric_side, preprocess_time, predict_time = side_mixup_detection(cropped_image, 64, model, label_encoder)
    fabric_side = fabric_side[0]
    print(f"Fabric side detection result: {fabric_side}")
    print(f"Preprocessing time: {preprocess_time:.6f} seconds")
    print(f"Prediction time: {predict_time:.6f} seconds")

    return fabric_side
