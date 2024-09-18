import cv2 as cv

from skimage.filters import sobel
from skimage.filters import roberts, sobel, scharr, prewitt
import pandas as pd
from scipy import ndimage as nd
import numpy as np 


def resizer(img,size):
    input_image_width,input_image_height,_ =img.shape
        
    if(input_image_width<input_image_height): #landscape
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE) 

    input_image_width,input_image_height,_ =img.shape
    resize_factor = input_image_height/size

    img = cv.resize(img, (int(input_image_height/resize_factor),int((input_image_width/input_image_height)*(input_image_height/resize_factor))))

    return img


def feature_extractor(dataset,imgc):
    x_train = dataset
    image_dataset = []  # Use a list to accumulate data
    for image in range(x_train.shape[0]):  # Iterate through each file
        input_img = x_train[image, :, :]
        img = input_img

        # Create a dictionary to hold the features
        feature_dict = {}

        # FEATURE 1: Pixel values
        feature_dict['Pixel_Value'] = img.reshape(-1)
        
        ksize = 30
        """
        # FEATURE 2: Gabor filter responses
        theta_min = 20
        theta_max = 30
        theta_factor = 16  # theta = theta/theta_factor * np.pi
        sigma_values = 1, 3
        lamda_factor = 8
        gamma_values = 1.3, 1.4, 1.5
        num = 1
        for theta in range(theta_min, theta_max):
            theta = (theta / theta_factor) * np.pi
            for sigma in sigma_values:
                for lamda in np.arange(0, np.pi, np.pi / lamda_factor):
                    for gamma in gamma_values:
                        gabor_label = 'Gabor' + str(num)
                        kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv.CV_32F)
                        fimg = cv.filter2D(img, cv.CV_8UC3, kernel)
                        feature_dict[gabor_label] = fimg.reshape(-1)
                        num += 1
        """


        kernel5 = cv.getGaborKernel((ksize, ksize), 1 , 4.516039439535327 , 1.1780972450961724 , 1.5 ,0, ktype=cv.CV_32F)
        fimg = cv.filter2D(img, cv.CV_8UC3, kernel5)
        feature_dict["kernel5"] = fimg.reshape(-1)
        kernel6 = cv.getGaborKernel((ksize, ksize), 1 , 4.516039439535327 , 1.1780972450961724 , 1.4 ,0, ktype=cv.CV_32F)
        fimg = cv.filter2D(img, cv.CV_8UC3, kernel6)
        feature_dict["kernel6"] = fimg.reshape(-1)
        kernel7 = cv.getGaborKernel((ksize, ksize), 1 , 4.516039439535327 , 1.1780972450961724 , 1.3 ,0, ktype=cv.CV_32F)
        fimg = cv.filter2D(img, cv.CV_8UC3, kernel7)
        feature_dict["kernel7"] = fimg.reshape(-1)


        # FEATURE 3: Canny edge detection
        img = cv.convertScaleAbs(img)
        edges = cv.Canny(img, 100, 200)
        feature_dict['Canny'] = edges.reshape(-1)

        # Add other features
        feature_dict['Sobel'] = sobel(img).reshape(-1)
        feature_dict['Scharr'] = scharr(img).reshape(-1)
        feature_dict['Prewitt'] = prewitt(img).reshape(-1)
        feature_dict['Gaussian_7'] = nd.gaussian_filter(img, sigma=7).reshape(-1)
        feature_dict['Roberts'] = roberts(img).reshape(-1)
        feature_dict['Gaussian_3'] = nd.gaussian_filter(img, sigma=3).reshape(-1)
        feature_dict['Median_3'] = nd.median_filter(img, size=3).reshape(-1)

        # HSV color channels (assuming imgc is the color version)
        hsv = cv.cvtColor(imgc, cv.COLOR_BGR2HSV)
        feature_dict['b_channel'] = imgc[:, :, 0].reshape(-1)
        feature_dict['g_channel'] = imgc[:, :, 1].reshape(-1)
        feature_dict['r_channel'] = imgc[:, :, 2].reshape(-1)
        feature_dict['h_channel'] = hsv[:, :, 0].reshape(-1)
        feature_dict['s_channel'] = hsv[:, :, 1].reshape(-1)
        feature_dict['v_channel'] = hsv[:, :, 2].reshape(-1)

        # Append the feature dictionary to the list
        image_dataset.append(pd.DataFrame(feature_dict))

    # Concatenate all the individual DataFrames into one large DataFrame
    image_dataset = pd.concat(image_dataset, ignore_index=True)
    return image_dataset
