import cv2 as cv
import numpy as np
from skimage.filters import roberts, sobel, scharr, prewitt
import pandas as pd
from scipy import ndimage as nd

def detection_filters(img, df,imgc,output_file="gabor_kernels.txt"):
    
    results = {}
    ksize = 3
    '''
    theta_min = 20
    theta_max = 30
    theta_factor = 16  # theta = theta/theta_factor * np.pi
    sigma_values = 1,3
    lamda_factor = 8
    gamma_values = 1.3, 1.4, 1.5
    ksize = 10
    
    '''
    """
    theta_iterations= 8
    theta_factor = 8  # theta = theta/theta_factor * np.pi
    sigma_values = 1,3
    lamda_factor = 8
    gamma_values = 1.3,1.4,1.5
    ksize_min = 0
    ksize_max = 9

    num = 1  # To count numbers up in order to give Gabor features a label in the data frame
    with open(output_file, 'w') as file:
        file.write('\n\n\n\nNew itwration\n\n\n\n')
        for ksize in range(ksize_min,ksize_max):
            # Number of thetas
            for theta in range(0, theta_iterations):
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
                            kernel_name = 'kernel_' + str(num)+'saved'
                            # Generating gabor kernel with the specific values
                            kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv.CV_32F)    
                            # Apply the gabor kernel to the training image. fmg = filtered 2D image 
                            fimg = cv.filter2D(img, cv.CV_8UC3, kernel)
                            # Reshape the 2D filtered image to a column
                            filtered_img = fimg.reshape(-1)
                            # Store the filtered image pixels in the dictionary
                            results[gabor_label] = filtered_img
                            # Print parameters used in the gabor kernel
                            #print(gabor_label, 'ksize',ksize,': sigma=', sigma, ': theta=', theta, ': lamda=', lamda, ': gamma=', gamma)
                            print(kernel_name,'= cv.getGaborKernel((',ksize,',',ksize,'),',sigma,',',theta,',',lamda,',',gamma,',0,','ktype=cv.CV_32F)#',gabor_label)
                            kernel_definition = (f"{kernel_name} = cv.getGaborKernel(({ksize}, {ksize}), {sigma}, {theta},{lamda}, {gamma}, 0, ktype=cv.CV_32F)  # {gabor_label}")
                            file.write(kernel_definition + '\n')
                            num += 1

    """
    
      
    kernel10 =  cv.getGaborKernel((ksize, ksize), 1 , 4.9269908169872414 , 1.1780972450961724*1.01 , 1.3 ,0, ktype=cv.CV_32F)
   
    
    #kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv.CV_32F) 
    #kernel1 = cv.getGaborKernel((ksize, ksize), 1 , 4.897787143782138 , 1.1780972450961724 , 1.3 ,0, ktype=cv.CV_32F)   
    kernel2 = cv.getGaborKernel((ksize, ksize), 1 , 4.897787143782138 , 1.1780972450961724 , 1.4 ,0, ktype=cv.CV_32F)
    kernel3 = cv.getGaborKernel((ksize, ksize), 1 , 4.71238898038469 , 1.1780972450961724 , 1.3 ,0, ktype=cv.CV_32F)
    #kernel4 = cv.getGaborKernel((ksize, ksize), 1 , 4.71238898038469 , 1.1780972450961724 , 1.5 ,0, ktype=cv.CV_32F)
    #kernel5 = cv.getGaborKernel((ksize, ksize), 1 , 4.516039439535327 , 1.1780972450961724 , 1.5 ,0, ktype=cv.CV_32F)
    kernel6 = cv.getGaborKernel((ksize, ksize), 1 , 4.516039439535327 , 1.1780972450961724 , 1.4 ,0, ktype=cv.CV_32F)
    #kernel7 = cv.getGaborKernel((ksize, ksize), 1 , 4.516039439535327 , 1.1780972450961724 , 1.3 ,0, ktype=cv.CV_32F)
    kernel8 = cv.getGaborKernel((ksize, ksize), 1 , 4.9269908169872414 , 1.1780972450961724 , 1.3 ,0, ktype=cv.CV_32F)
    kernel9 = cv.getGaborKernel((ksize, ksize), 1 , 4.9269908169872414 , 1.1780972450961724 , 1.4 ,0, ktype=cv.CV_32F)

   #results["gabor_1"] = (cv.filter2D(img, cv.CV_8UC3, kernel1)).reshape(-1)
    results["gabor_2"] = (cv.filter2D(img, cv.CV_8UC3, kernel2)).reshape(-1)
    results["gabor_3"] = (cv.filter2D(img, cv.CV_8UC3, kernel3)).reshape(-1)
    #results["gabor_4"] = (cv.filter2D(img, cv.CV_8UC3, kernel4)).reshape(-1)
    #results["gabor_5"] = (cv.filter2D(img, cv.CV_8UC3, kernel5)).reshape(-1)
    results["gabor_6"] = (cv.filter2D(img, cv.CV_8UC3, kernel6)).reshape(-1)
   # results["gabor_7"] = (cv.filter2D(img, cv.CV_8UC3, kernel7)).reshape(-1)
    results["gabor_8"] = (cv.filter2D(img, cv.CV_8UC3, kernel8)).reshape(-1)
    results["gabor_9"] = (cv.filter2D(img, cv.CV_8UC3, kernel9)).reshape(-1)


    results["gabor_10"] = (cv.filter2D(img, cv.CV_8UC3, kernel10)).reshape(-1)

    """
    cv.imwrite(f"images/captured/kernel2.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel2)))
    cv.imwrite(f"images/captured/kernel3.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel3)))
    cv.imwrite(f"images/captured/kernel6.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel6)))
    cv.imwrite(f"images/captured/kernel8.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel8)))
    cv.imwrite(f"images/captured/kernel9.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel9)))
    cv.imwrite(f"images/captured/kernel10.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel10)))


    """




   # results["gabor_11"] = (cv.filter2D(img, cv.CV_8UC3, kernel11)).reshape(-1)
    #results["gabor_12"] = (cv.filter2D(img, cv.CV_8UC3, kernel12)).reshape(-1)
   # results["gabor_14"] = (cv.filter2D(img, cv.CV_8UC3, kernel14)).reshape(-1)
   # results["gabor_15"] = (cv.filter2D(img, cv.CV_8UC3, kernel15)).reshape(-1)
    kernel_2252saved = cv.getGaborKernel((5, 5), 3, 2.356194490192345,2.356194490192345, 1.4, 0, ktype=cv.CV_32F)  # Gabor2252
    kernel_1869saved = cv.getGaborKernel((4, 4), 3, 2.356194490192345,2.356194490192345, 1.5, 0, ktype=cv.CV_32F)  # Gabor1869
    kernel_2157saved = cv.getGaborKernel((5, 5), 3, 1.5707963267948966,2.356194490192345, 1.5, 0, ktype=cv.CV_32F)  # Gabor2157
    kernel_1676saved = cv.getGaborKernel((4, 4), 3, 0.7853981633974483,2.356194490192345, 1.4, 0, ktype=cv.CV_32F)  # Gabor1676
    
    results["kernel_2252saved"] = (cv.filter2D(img, cv.CV_8UC3, kernel_2252saved)).reshape(-1)
    results["kernel_1869saved"] = (cv.filter2D(img, cv.CV_8UC3, kernel_1869saved)).reshape(-1)
    results["kernel_2157saved"] = (cv.filter2D(img, cv.CV_8UC3, kernel_2157saved)).reshape(-1)
    results["kernel_1676saved"] = (cv.filter2D(img, cv.CV_8UC3, kernel_1676saved)).reshape(-1)

    kernel_694saved = cv.getGaborKernel((1, 1), 1, 2.356194490192345,2.748893571891069, 1.3, 0, ktype=cv.CV_32F)  # Gabor694
    kernel_591saved = cv.getGaborKernel((1, 1), 1, 1.5707963267948966,1.5707963267948966, 1.5, 0, ktype=cv.CV_32F)  # Gabor591
    kernel_1018saved = cv.getGaborKernel((2, 2), 1, 1.9634954084936207,1.1780972450961724, 1.3, 0, ktype=cv.CV_32F)  # Gabor1018
    kernel_571saved = cv.getGaborKernel((1, 1), 3, 1.1780972450961724,2.356194490192345, 1.3, 0, ktype=cv.CV_32F)  # Gabor571
    kernel_1980saved = cv.getGaborKernel((5, 5), 1, 0.39269908169872414,1.1780972450961724, 1.5, 0, ktype=cv.CV_32F)  # Gabor1980
    
    results["kernel_694saved"] = (cv.filter2D(img, cv.CV_8UC3, kernel_694saved)).reshape(-1)
    results["kernel_591saved"] = (cv.filter2D(img, cv.CV_8UC3, kernel_591saved)).reshape(-1)
    results["kernel_1018saved"] = (cv.filter2D(img, cv.CV_8UC3, kernel_1018saved)).reshape(-1)
    results["kernel_571saved"] = (cv.filter2D(img, cv.CV_8UC3, kernel_571saved)).reshape(-1)
    results["kernel_1980saved"] = (cv.filter2D(img, cv.CV_8UC3, kernel_1980saved)).reshape(-1)

    """
    cv.imwrite(f"images/captured/kernel_2252saved.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel_2252saved)))
    cv.imwrite(f"images/captured/kernel_1869saved.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel_1869saved)))
    cv.imwrite(f"images/captured/kernel_2157saved.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel_2157saved)))
    cv.imwrite(f"images/captured/kernel_1676saved.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel_1676saved)))
    
    cv.imwrite(f"images/captured/kernel_694saved.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel_694saved)))
    cv.imwrite(f"images/captured/kernel_591saved.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel_591saved)))
    cv.imwrite(f"images/captured/kernel_1018saved.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel_1018saved)))
    cv.imwrite(f"images/captured/kernel_571saved.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel_571saved)))
    cv.imwrite(f"images/captured/kernel_1980saved.jpg",(cv.filter2D(img, cv.CV_8UC3, kernel_1980saved)))
    """

    


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

    #cv.imwrite(f"images/captured/gaussian_img.jpg",gaussian_img)

    # Apply Median with size=3
    median_img = nd.median_filter(img, size=3)
    ##cv.imshow("median_img", median_img)
    results['Median s3'] = median_img.reshape(-1)
    
    #cv.imwrite(f"images/captured/median_img.jpg", median_img)

    hsv = cv.cvtColor(imgc, cv.COLOR_BGR2HSV)
    # Extract the saturatin channel
    b_channel = imgc[:, :, 0]
    g_channel = imgc[:, :, 1]
    r_channel = imgc[:, :, 2]
    #h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    """
    cv.imwrite(f"images/captured/b_channel.jpg",b_channel)
    cv.imwrite(f"images/captured/g_channel.jpg",g_channel)
    cv.imwrite(f"images/captured/r_channel.jpg",r_channel)
    cv.imwrite(f"images/captured/s_channel.jpg",s_channel)
    cv.imwrite(f"images/captured/v_channel.jpg",v_channel)
    
    """
    results['b_channel'] = b_channel.reshape(-1)
    results['g_channel'] = g_channel.reshape(-1)
    results['r_channel'] = r_channel.reshape(-1)
    #results['h_channel'] = h_channel.reshape(-1)
    results['s_channel'] = s_channel.reshape(-1)
    results['v_channel'] = v_channel.reshape(-1)

    # Combine the existing DataFrame with the new data
    df = pd.concat([df, pd.DataFrame(results)], axis=1)
    
    return df
