import cv2 as cv
import numpy as np
from skimage.filters import roberts, sobel, scharr, prewitt
import pandas as pd
from scipy import ndimage as nd

def detection_filters(img, df,imgc):
    '''
    theta_min = 20
    theta_max = 30
    theta_factor = 16  # theta = theta/theta_factor * np.pi
    sigma_values = 1,3
    lamda_factor = 4
    gamma_values = 1.3, 1.4, 1.5
    ksize = 10
    
    '''
    theta_min = 20
    theta_max = 30
    theta_factor = 16  # theta = theta/theta_factor * np.pi
    sigma_values = 1,3
    lamda_factor = 8
    gamma_values = 1.3, 1.4, 1.5
    ksize = 3

    num = 1  # To count numbers up in order to give Gabor features a label in the data frame
    results = {}
    """
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
                    print(gabor_label, ': sigma=', sigma, ': theta=', theta, ': lamda=', lamda, ': gamma=', gamma)
                    print('kernel = cv.getGaborKernel((ksize,ksize),',sigma,',',theta,',',lamda,',',gamma,',0,','ktype=cv.CV_32F)#',gabor_label)
                    num += 1
    """
    
    kernel1 = cv.getGaborKernel((3, 3), 0.1, 0.628319, 3, 0.08, 0.188496, ktype=cv.CV_32F) 
    kernel2 = cv.getGaborKernel((3, 3), 0.1, 4.08407, 3, 0.08, 0, ktype=cv.CV_32F) 
    kernel3 = cv.getGaborKernel((3, 3), 0.2, 0.628319, 0.4, 0.08, 0.251327, ktype=cv.CV_32F) 
    kernel4 = cv.getGaborKernel((3, 3), 0.2, 0.942478, 0.4, 0.08, 0, ktype=cv.CV_32F)  
    kernel5 = cv.getGaborKernel((3, 3), 0.2, 3.76911, 0.4, 0.08, 0, ktype=cv.CV_32F) 
    kernel6 = cv.getGaborKernel((3, 3), 0.2, 4.08407, 0.4, 0.04, 0.188496, ktype=cv.CV_32F) 
      
   
    #Gabor311 : theta= 5.105088062083414 : sigma= 1 : lamda= 2.748893571891069 : gamma= 0.5
    #kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv.CV_32F) 
    #kernel1 = cv.getGaborKernel((ksize, ksize), 1 , 4.897787143782138 , 1.1780972450961724 , 1.3 ,0, ktype=cv.CV_32F)   
    #kernel2 = cv.getGaborKernel((ksize, ksize), 1 , 4.897787143782138 , 1.1780972450961724 , 1.4 ,0, ktype=cv.CV_32F)
    #kernel3 = cv.getGaborKernel((ksize, ksize), 1 , 4.71238898038469 , 1.1780972450961724 , 1.3 ,0, ktype=cv.CV_32F)
    #kernel4 = cv.getGaborKernel((ksize, ksize), 1 , 4.71238898038469 , 1.1780972450961724 , 1.5 ,0, ktype=cv.CV_32F)
    #kernel5 = cv.getGaborKernel((ksize, ksize), 1 , 4.516039439535327 , 1.1780972450961724 , 1.5 ,0, ktype=cv.CV_32F)
    #kernel6 = cv.getGaborKernel((ksize, ksize), 1 , 4.516039439535327 , 1.1780972450961724 , 1.4 ,0, ktype=cv.CV_32F)
    kernel7 = cv.getGaborKernel((ksize, ksize), 1 , 4.516039439535327 , 1.1780972450961724 , 1.3 ,0, ktype=cv.CV_32F)
    kernel8 = cv.getGaborKernel((ksize, ksize), 1 , 4.9269908169872414 , 1.1780972450961724 , 1.3 ,0, ktype=cv.CV_32F)
    kernel9 = cv.getGaborKernel((ksize, ksize), 1 , 4.9269908169872414 , 1.1780972450961724 , 1.4 ,0, ktype=cv.CV_32F)

    results["gabor_1"] = (cv.filter2D(img, cv.CV_8UC3, kernel1)).reshape(-1)
    results["gabor_2"] = (cv.filter2D(img, cv.CV_8UC3, kernel2)).reshape(-1)
    results["gabor_3"] = (cv.filter2D(img, cv.CV_8UC3, kernel3)).reshape(-1)
    results["gabor_4"] = (cv.filter2D(img, cv.CV_8UC3, kernel4)).reshape(-1)
    results["gabor_5"] = (cv.filter2D(img, cv.CV_8UC3, kernel5)).reshape(-1)
    results["gabor_6"] = (cv.filter2D(img, cv.CV_8UC3, kernel6)).reshape(-1)
    results["gabor_7"] = (cv.filter2D(img, cv.CV_8UC3, kernel7)).reshape(-1)
    results["gabor_8"] = (cv.filter2D(img, cv.CV_8UC3, kernel8)).reshape(-1)
    results["gabor_9"] = (cv.filter2D(img, cv.CV_8UC3, kernel9)).reshape(-1)

   

    #"""
    # Apply Canny
    #edges = cv.Canny(img, 100, 200)
    ##cv.imshow("edges", edges)
    #results['Canny Edge'] = edges.reshape(-1)


    # Apply Sobel
    #edge_sobel = sobel(img)
    ##cv.imshow("edge_sobel", edge_sobel)
    #results['Sobel'] = edge_sobel.reshape(-1)

    # Apply Scharr
    #edge_scharr = scharr(img)
    ##cv.imshow("edge_scharr", edge_scharr)
    #results['Scharr'] = edge_scharr.reshape(-1)

    # Apply Prewitt
    #edge_prewitt = prewitt(img)
    ##cv.imshow("edge_prewitt", edge_prewitt)
    #results['Prewitt'] = edge_prewitt.reshape(-1)
    
    # Apply Gaussian with sigma=7
    #gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    ##cv.imshow("gaussian_img2", gaussian_img2)
    #results['Gaussian s7'] = gaussian_img2.reshape(-1)
    #"""
    # Apply Roberts edge
    #edge_roberts = roberts(img)
    ##cv.imshow("edge_roberts", edge_roberts)
    #results['Roberts'] = edge_roberts.reshape(-1) 

    # Apply Gaussian with sigma=3
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    ##cv.imshow("gaussian_img", gaussian_img)
    results['Gaussian s3'] = gaussian_img.reshape(-1)


    # Apply Median with size=3
    median_img = nd.median_filter(img, size=3)
    ##cv.imshow("median_img", median_img)
    results['Median s3'] = median_img.reshape(-1)

    hsv = cv.cvtColor(imgc, cv.COLOR_BGR2HSV)
    # Extract the saturatin channel
    b_channel = imgc[:, :, 0]
    g_channel = imgc[:, :, 1]
    r_channel = imgc[:, :, 2]
    #h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    results['b_channel'] = b_channel.reshape(-1)
    results['g_channel'] = g_channel.reshape(-1)
    results['r_channel'] = r_channel.reshape(-1)
    #results['h_channel'] = h_channel.reshape(-1)
    results['s_channel'] = s_channel.reshape(-1)
    results['v_channel'] = v_channel.reshape(-1)

    # Combine the existing DataFrame with the new data
    df = pd.concat([df, pd.DataFrame(results)], axis=1)
    
    return df
