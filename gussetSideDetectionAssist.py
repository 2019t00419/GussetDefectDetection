import numpy as np
import cv2 as cv
import pickle
import time
from gussetSideDetectionFilters import resizer, feature_extractor

# Load the trained model and label encoder (no need to load each time for multiple predictions)
model = pickle.load(open("detectionSupportModelforSide", 'rb'))
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

def detect_gusset_side(imgc, size, model, label_encoder):
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
image_path = "test/Classification/Test_images/new (8).jpg"
prediction, preprocess_time, predict_time = detect_gusset_side(image_path, 240, model, label_encoder)

# Print the results
print(f"The predicted class for the image is: {prediction[0]}")
print(f"Preprocessing time: {preprocess_time:.6f} seconds")
print(f"Prediction time: {predict_time:.6f} seconds")
"""