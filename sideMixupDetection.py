import numpy as np
import cv2 as cv
import pickle
import time
from sideMixupDetectionFilters import resizer, feature_extractor

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

def crop_image(original_frame, longest_contour, count):
    M = cv.moments(longest_contour)
    if M['m00'] == 0:
        return None 
    else:
        frame_height, frame_width, channels = original_frame.shape
        print(original_frame.shape)
        cx = int(frame_width * (M['m10'] / M['m00']) / 960)
        cy = int(frame_height * (M['m01'] / M['m00']) / 1280)

        print("Center point = (" + str(cx) + "," + str(cy) + ")")

        # Define the coordinates
        tlx, tly = cx - int(image_width/2), cy - int(image_width/2)  # Top-left corner
        brx, bry = cx + int(image_width/2), cy + int(image_width/2)  # Bottom-right corner

        print("Top left point = (" + str(tlx) + "," + str(tly) + ")")
        print("Bottom right point = (" + str(brx) + "," + str(bry) + ")")

        # Ensure the coordinates define a square area
        if abs(tlx - brx) != abs(tly - bry):
            raise ValueError("The provided coordinates do not define a square area.")

        # Crop the image
        cropped_image = original_frame[tly:bry, tlx:brx]
        grayscale_cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

        # Display the cropped image
        ##cv.imshow("Otsu cropped Image", otsu_cropped_image)
        cv.imwrite("images/out/cropped/cropped (" + str(count) + ").jpg", grayscale_cropped_image)
        #fabric_side = detect_side(otsu_cropped_image)

        fabric_side, preprocess_time, predict_time = side_mixup_detection(cropped_image, 64, model, label_encoder)
        fabric_side = fabric_side[0]
        
        # Print the results
        print(f"The fabric is: {fabric_side}")
        print(f"Preprocessing time: {preprocess_time:.6f} seconds")
        print(f"Prediction time: {predict_time:.6f} seconds")
    return fabric_side
