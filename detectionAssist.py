
import numpy as np
import cv2 as cv
import pandas as pd
from detectionAssistfilters import detection_filtes
import pickle


def feature_extraction(input_img):
    ##cv.imshow("original image", input_img)
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv.cvtColor(input_img,cv.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("The module works only with grayscale and RGB images!")

    #img = cv.GaussianBlur(img, (5, 5), 0)
    print(img)    
    #Save original image pixels into a data frame. This is our Feature #1.
    img2 = img.reshape(-1)
    df = pd.DataFrame()
    df['Original Image'] = img2
    print(df)
    
    #Generate Gabor features
   
    df = detection_filtes(img,df,input_img)

    return img,df,input_img

import time

def detection_support(image):
    total_start_time = time.time()  # Start total time for the function
    
    # Initial preparation
    prep_start_time = time.time()
    detection_height = 720
    input_image_width, input_image_height, _ = image.shape
    if input_image_width < input_image_height:  # Landscape
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    input_image_width, input_image_height, _ = image.shape
    resize_factor = input_image_height / detection_height
    image = cv.resize(image, (int(input_image_height / resize_factor), int((input_image_width / input_image_height) * (input_image_height / resize_factor))))
    prep_end_time = time.time()
    print(f"Time taken for initial preparation: {prep_end_time - prep_start_time:.6f} seconds")

    # Model loading
    model_start_time = time.time()
    filename = "detectionSupportModel"
    loaded_model = pickle.load(open(filename, 'rb'))
    model_end_time = time.time()
    print(f"Time taken to load model: {model_end_time - model_start_time:.6f} seconds")

    # Feature extraction
    feature_extraction_start_time = time.time()
    img, X, image = feature_extraction(image)
    feature_extraction_end_time = time.time()
    print(f"Time taken for feature extraction: {feature_extraction_end_time - feature_extraction_start_time:.6f} seconds")

    # Prediction and segmentation
    prediction_start_time = time.time()
    result = loaded_model.predict(X)
    segmented = result.reshape((img.shape))
    segmented = segmented.astype(np.int8)
    #probabilities = loaded_model.predict_proba(X)
    #predicted_classes = np.argmax(probabilities, axis=1)
    #confidences = np.max(probabilities, axis=1)
    #average_confidence = np.mean(confidences)
    #print(f"Average confidence of the segmentation: {average_confidence:.2f}")
    prediction_end_time = time.time()
    print(f"Time taken for prediction and segmentation: {prediction_end_time - prediction_start_time:.6f} seconds")

    # Image conversion and scaling
    conversion_start_time = time.time()
    segmented_8u = cv.convertScaleAbs(segmented)
    support_image = np.zeros_like(segmented_8u)
    image[segmented_8u == 0] = [0, 0, 255]  # BGR
    image[segmented_8u == 1] = [0, 255, 0]  # BGR
    image[segmented_8u == 2] = [255, 0, 0]  # BGR
    image[segmented_8u == 3] = [255, 255, 255]  # BGR
    support_image[segmented_8u == 3] = [255]
    conversion_end_time = time.time()
    print(f"Time taken for image conversion and scaling: {conversion_end_time - conversion_start_time:.6f} seconds")

    # Resizing and saving the segmented image
    resizing_start_time = time.time()
    resized_support_image = cv.resize(support_image, (input_image_height, input_image_width))
    resized_image = cv.resize(image, (input_image_height, input_image_width))
    cv.imwrite('test/Segmanted_images/segmented_bgr_image.jpg', resized_image)
    resizing_end_time = time.time()
    print(f"Time taken for resizing and saving: {resizing_end_time - resizing_start_time:.6f} seconds")

    total_end_time = time.time()
    print(f"Total time taken for detection_support function: {total_end_time - total_start_time:.6f} seconds")

    return resized_support_image, resized_image


image_path = 'test/Test_images/captured_20241031_1638130.jpg'
image = cv.imread(image_path)

binary_image, processed_image = detection_support(image)
print(type(processed_image), processed_image.shape if processed_image is not None else "None")


if processed_image is not None:
    
    resized_image1 = cv.resize(processed_image, (360, 640))
    cv.imshow("view", resized_image1)
else:
    print("Error: `detection_support` returned None.")


cv.waitKey(0)
cv.destroyAllWindows()
