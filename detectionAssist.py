
import numpy as np
import cv2 as cv
import pandas as pd
from detectionAssistfilters import detection_filtes
import pickle


def feature_extraction(input_img):
    #cv.imshow("original image", input_img)
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


def detection_support(image):
    
    
    detection_height = 720 #detection width is set for portrait images
    print(image.shape)
    input_image_width,input_image_height,_ =image.shape
    if(input_image_width<input_image_height): #landscape
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE) 
    
    input_image_width,input_image_height,_ =image.shape
    resize_factor = input_image_height/detection_height

    #print (input_image_height,',',input_image_width)
    #print (input_image_height/resize_factor,',',(input_image_width/input_image_height)*(input_image_height/resize_factor))
   
    image = cv.resize(image, (int(input_image_height/resize_factor),int((input_image_width/input_image_height)*(input_image_height/resize_factor))))

    #image_path = 'test/Test_images/test_image_1.jpg'
    #image = cv.imread(image_path)
    filename = "detectionSupportModel"
    #filename = "sandstone_model_multi_image"
    loaded_model = pickle.load(open(filename, 'rb'))

    #cv.imshow("original image", image)
    #image = cv.GaussianBlur(image, (9, 9), 0)

    #Call the feature extraction function.
    img,X,image = feature_extraction(image)
    result = loaded_model.predict(X)
    segmented = result.reshape((img .shape))
    segmented = segmented.astype(np.int8)

    # Predict the probabilities for each class
    probabilities = loaded_model.predict_proba(X)

    # Get the predicted classes (i.e., the segmentation result)
    predicted_classes = np.argmax(probabilities, axis=1)

    # Get the confidence for the predicted class (max probability for each pixel)
    confidences = np.max(probabilities, axis=1)

    # Calculate the average confidence across all pixels
    average_confidence = np.mean(confidences)
    print(f"Average confidence of the segmentation: {average_confidence:.2f}")

    # Alternatively, use convertScaleAbs
    segmented_8u = cv.convertScaleAbs(segmented)
    support_image = np.zeros_like(segmented_8u)

    segmented_bgr_image = np.zeros((segmented_8u.shape[0], segmented_8u.shape[1], 3), dtype=np.uint8)

    # Convert values back to BGR
    #image[segmented_8u == 0] = [0, 0, 255]  # Red
    image[segmented_8u == 1] = [0, 255, 0]  # Green for adhesive
    #segmented_bgr_image[segmented_8u == 1] = [0, 255, 0]  # Green for adhesive
    #segmented_bgr_image[segmented_8u == 2] = [255, 0, 0]  # Blue
    
    support_image[segmented_8u == 1] = [255]  # Red

    cv.imshow("viesdw",image)
    resized_image = cv.resize(support_image, (input_image_height,input_image_width))
    cv.imwrite('test/Segmanted_images/segmented_bgr_image.jpg', resized_image)

    #cv.imshow("view",support_image)

    return resized_image
'''
image_path = 'test/Test_images/captured (13).jpg'
image = cv.imread(image_path)

cv.imshow("view",detection_support(image))

cv.waitKey(0)
cv.destroyAllWindows()
'''