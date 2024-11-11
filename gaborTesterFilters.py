import cv2 as cv
import numpy as np
import pandas as pd
from scipy import ndimage as nd

def detection_filters(img, df, imgc):
    # Gabor filter parameters and factors
    sigma_factor = 0.5
    theta_factor = np.pi / 4
    lamda_factor = np.pi / 4
    gamma_factor = 0.5
    ksize = 3

    # Calculate the number of values for each parameter
    sigma_values_count = int(1 / sigma_factor)  # Since sigma ranges from 0 to 1
    theta_values_count = int((2 * np.pi) / theta_factor)  # Theta ranges from 0 to 2*pi
    lamda_values_count = int(np.pi / lamda_factor)  # Lambda ranges from 0 to pi
    gamma_values_count = int(2 / gamma_factor)  # Gamma ranges from 0 to 2

    # Calculate the total number of combinations
    total_combinations = sigma_values_count * theta_values_count * lamda_values_count * gamma_values_count
    print("Total number of combinations:", total_combinations)

    # Initialize variables for filter creation
    num = 0
    results = {}

    # Iterate through combinations of filters
    for sigma in np.arange(0, 1, sigma_factor):
        for theta in np.arange(0, 2 * np.pi, theta_factor):  # Iterating through theta values
            for lamda in np.arange(0, np.pi, lamda_factor):  # Iterating through wavelength (lambda)
                for gamma in np.arange(0, 2, gamma_factor):  # Iterating through gamma values
                    # Generate the Gabor kernel for each combination
                    gabor_label = f'Gabor_{num}'
                    kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv.CV_32F)
                    # Apply the kernel to the image and reshape the result to a vector
                    fimg = cv.filter2D(img, cv.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    results[gabor_label] = filtered_img
                    num += 1
                    
    # Add other filters (Gaussian, Median, Channels) to results
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    results['Gaussian_s3'] = gaussian_img.reshape(-1)
    median_img = nd.median_filter(img, size=3)
    results['Median_s3'] = median_img.reshape(-1)

    hsv = cv.cvtColor(imgc, cv.COLOR_BGR2HSV)
    b_channel, g_channel, r_channel = imgc[:, :, 0], imgc[:, :, 1], imgc[:, :, 2]
    s_channel, v_channel = hsv[:, :, 1], hsv[:, :, 2]

    results['b_channel'] = b_channel.reshape(-1)
    results['g_channel'] = g_channel.reshape(-1)
    results['r_channel'] = r_channel.reshape(-1)
    results['s_channel'] = s_channel.reshape(-1)
    results['v_channel'] = v_channel.reshape(-1)

    # Convert results dictionary to a DataFrame and append to the existing DataFrame
    df = pd.concat([df, pd.DataFrame(results)], axis=1)

    # Export the DataFrame to an Excel file
    df.to_excel('gabor_filter_results.xlsx', index=False)

    return df
