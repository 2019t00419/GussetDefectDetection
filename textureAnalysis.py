
import numpy as np
import cv2 as cv
import pandas as pd
from filter_generation import filter_generation
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
   
    df = filter_generation(img,df,input_img)

    return img,df,input_img


def detect_stains(image):
    resize_factor = 4.5
    input_image_width,input_image_height,_ =image.shape
    
    #print (input_image_height,',',input_image_width)
    #print (input_image_height/resize_factor,',',(input_image_width/input_image_height)*(input_image_height/resize_factor))
   
    image = cv.resize(image, (int(input_image_height/resize_factor),int((input_image_width/input_image_height)*(input_image_height/resize_factor))))

    #image_path = 'test/Test_images/test_image_1.jpg'
    #image = cv.imread(image_path)
    filename = "fabric_defect_model_multi_image"
    #filename = "sandstone_model_multi_image"
    loaded_model = pickle.load(open(filename, 'rb'))

    ##cv.imshow("original image", image)
    #image = cv.GaussianBlur(image, (9, 9), 0)

    #Call the feature extraction function.
    img,X,image = feature_extraction(image)
    result = loaded_model.predict(X)
    segmented = result.reshape((img .shape))
    segmented = segmented.astype(np.int8)

    # Alternatively, use convertScaleAbs
    segmented_8u = cv.convertScaleAbs(segmented)

    segmented_bgr_image = np.zeros((segmented_8u.shape[0], segmented_8u.shape[1], 3), dtype=np.uint8)

    # Convert values back to BGR
    #image[segmented_8u == 0] = [0, 0, 255]  # Red
    image[segmented_8u == 1] = [0, 255, 0]  # Green
    #segmented_bgr_image[segmented_8u == 2] = [255, 0, 0]  # Blue

    #cv.imshow("segmented_bgr_image", image)

    cv.imwrite('test/Segmanted_images/segmented_bgr_image.jpg', segmented_bgr_image)

    stain_marks = True

    cv.waitKey(0)
    cv.destroyAllWindows()

    return stain_marks


#image_path = 'test/Test_images/T (20).jpg'
#image = cv.imread(image_path)

#stain_marks = detect_stains(image)
#print("Stain marks status : " ,stain_marks)