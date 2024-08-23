import cv2 as cv
import numpy as np
from skimage.filters import roberts, sobel, scharr, prewitt
import pandas as pd
from scipy import ndimage as nd

def filter_generation(img, df):
    theta_min = 20
    theta_max = 30
    theta_factor = 16  # theta = theta/theta_factor * np.pi
    sigma_values = 1,3
    lamda_factor = 8
    gamma_values = 0.05, 0.5, 0.1
    ksize = 30

    num = 1  # To count numbers up in order to give Gabor features a label in the data frame
    results = {}

    # Number of thetas
    for theta in range(theta_min, theta_max):
        # Calculating theta for each iteration. max 2pi
        theta = (theta / theta_factor) * np.pi
        # Sigma values list
        for sigma in sigma_values:
            # Range of wavelengths. 0 to pi with increments of pi/4
            for lamda in np.arange(0, np.pi, np.pi / lamda_factor):
                # Gamma values list
                for gamma in gamma_values:     
                    # Label for the gabor kernel  
                    gabor_label = 'Gabor' + str(num)
                    # Generating gabor kernel with the specific values
                    kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv.CV_32F)    
                    # Apply the gabor kernel to the training image. fmg = filtered 2D image 
                    fimg = cv.filter2D(img, cv.CV_8UC3, kernel)
                    # Reshape the 2D filtered image to a column
                    filtered_img = fimg.reshape(-1)
                    # Store the filtered image pixels in the dictionary
                    results[gabor_label] = filtered_img
                    # Print parameters used in the gabor kernel
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1

    # Apply Canny
    edges = cv.Canny(img, 100, 200)
    results['Canny Edge'] = edges.reshape(-1)

    # Apply Roberts edge
    edge_roberts = roberts(img)
    results['Roberts'] = edge_roberts.reshape(-1)

    # Apply Sobel
    edge_sobel = sobel(img)
    results['Sobel'] = edge_sobel.reshape(-1)

    # Apply Scharr
    edge_scharr = scharr(img)
    results['Scharr'] = edge_scharr.reshape(-1)

    # Apply Prewitt
    edge_prewitt = prewitt(img)
    results['Prewitt'] = edge_prewitt.reshape(-1)

    # Apply Gaussian with sigma=3
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    results['Gaussian s3'] = gaussian_img.reshape(-1)

    # Apply Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    results['Gaussian s7'] = gaussian_img2.reshape(-1)

    # Apply Median with size=3
    median_img = nd.median_filter(img, size=3)
    results['Median s3'] = median_img.reshape(-1)

    # Combine the existing DataFrame with the new data
    df = pd.concat([df, pd.DataFrame(results)], axis=1)
    
    return df
