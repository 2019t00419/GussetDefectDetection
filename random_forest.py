
import numpy as np
import cv2
import pandas as pd
 
img = cv2.imread('test/h1.jpg')
labeled_img = cv2.imread('test/hh0.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
cv2.imshow("original image", img)
img = cv2.GaussianBlur(img, (5, 5), 0)
print(img)
#Here, if you have multichannel image then extract the right channel instead of converting the image to grey. 
#For example, if DAPI contains nuclei information, extract the DAPI channel image first. 

#Multiple images can be used for training. For that, you need to concatenate the data

#Save original image pixels into a data frame. This is our Feature #1.
img2 = img.reshape(-1)
df = pd.DataFrame()
df['Original Image'] = img2
print(df)

#Generate Gabor features
num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
kernels = []
for theta in range(2):   #Define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1, 3):  #Sigma with 1 and 3
        for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
            for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5         
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
#                print(gabor_label)
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
                
########################################
#Gerate OTHER FEATURES and add them to the data frame
         
print(df)       
#CANNY EDGE
edges = cv2.Canny(img, 100,200)   #Image, min and max values
edges1 = edges.reshape(-1)
df['Canny Edge'] = edges1 #Add column to original dataframe

from skimage.filters import roberts, sobel, scharr, prewitt

#ROBERTS EDGE
edge_roberts = roberts(img)
edge_roberts1 = edge_roberts.reshape(-1)
df['Roberts'] = edge_roberts1

#SOBEL
edge_sobel = sobel(img)
edge_sobel1 = edge_sobel.reshape(-1)
df['Sobel'] = edge_sobel1

#SCHARR
edge_scharr = scharr(img)
edge_scharr1 = edge_scharr.reshape(-1)
df['Scharr'] = edge_scharr1

#PREWITT
edge_prewitt = prewitt(img)
edge_prewitt1 = edge_prewitt.reshape(-1)
df['Prewitt'] = edge_prewitt1

#GAUSSIAN with sigma=3
from scipy import ndimage as nd
gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian_img1 = gaussian_img.reshape(-1)
df['Gaussian s3'] = gaussian_img1

#GAUSSIAN with sigma=7
gaussian_img2 = nd.gaussian_filter(img, sigma=7)
gaussian_img3 = gaussian_img2.reshape(-1)
df['Gaussian s7'] = gaussian_img3

#MEDIAN with sigma=3
median_img = nd.median_filter(img, size=3)
median_img1 = median_img.reshape(-1)
df['Median s3'] = median_img1

#VARIANCE with size=3
#variance_img = nd.generic_filter(img, np.var, size=3)
#variance_img1 = variance_img.reshape(-1)
#df['Variance s3'] = variance_img1  #Add column to original dataframe
       
print(df)

######################################                

#Now, add a column in the data frame for the Labels
#For this, we need to import the labeled image
#Remember that you can load an image with partial labels 
#But, drop the rows with unlabeled data
print(labeled_img.shape)
print("labeled_img")
print(labeled_img)
cv2.imshow("labeled_img", labeled_img)


# Create an empty array for the output image with the same dimensions as the input
output_image = np.zeros((labeled_img.shape[0], labeled_img.shape[1]), dtype=np.uint8)

# Extract the blue, green, and red channels
blue_channel = labeled_img[:, :, 0]
green_channel = labeled_img[:, :, 1]
red_channel = labeled_img[:, :, 2]

# Apply conditions
output_image[(blue_channel > 0) & (green_channel == 0) & (red_channel == 0)] = 1
output_image[(red_channel > 0) & (blue_channel == 0) & (green_channel == 0)] = 2
output_image[(green_channel > 0) & (blue_channel == 0) & (red_channel == 0)] = 3


labeled_img1 = output_image.reshape(-1)
df['Labels'] = labeled_img1
print(df)

original_img_data = df.drop(labels = ["Labels"], axis=1) #Use for prediction
print(original_img_data)
#df.to_csv("Gabor.csv")
df = df[df.Labels != 0]
print(df)
#########################################################

#Define the dependent variable that needs to be predicted (labels)
Y = df["Labels"].values

#Encode Y values to 0, 1, 2, 3, .... (NOt necessary but makes it easy to use other tools like ROC plots)
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)


print(Y)

#Define the independent variables
X = df.drop(labels = ["Labels"], axis=1) 
print(X)

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


# Import the model we are using
#RandomForestRegressor is for regression type of problems. 
#For classification we use RandomForestClassifier.
#Both yield similar results except for regressor the result is float
#and for classifier it is an integer. 

from sklearn.ensemble import RandomForestClassifier
# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 20, random_state = 42)

# Train the model on training data
model.fit(X_train, y_train)

# Get numerical feature importances
#importances = list(model.feature_importances_)

#Let us print them into a nice format.

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)

prediction_RF = model.predict(X_test)
from sklearn import metrics
#Print the prediction accuracy
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy using Random Forest= ", metrics.accuracy_score(y_test, prediction_RF)*100,"%")

import pickle

#Save the trained model as pickle string to disk for future use
filename = "sandstone_model"
pickle.dump(model, open(filename, 'wb'))

#To test the model on future datasets
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(original_img_data)

segmented = result.reshape((img.shape))


# Normalize and convert to 8-bit unsigned integer
segmented_8u = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Alternatively, use convertScaleAbs
segmented_8u = cv2.convertScaleAbs(segmented)


segmented_bgr_image = np.zeros((segmented_8u.shape[0], segmented_8u.shape[1], 3), dtype=np.uint8)

# Convert values back to BGR
segmented_bgr_image[segmented_8u == 1] = [0, 0, 255]  # Blue
segmented_bgr_image[segmented_8u == 2] = [255, 0, 0]  # Red
segmented_bgr_image[segmented_8u == 3] = [0, 255, 0]  # Green

cv2.imshow("segmented_bgr_image", segmented_bgr_image)



blue_pixels = np.sum(segmented_8u == 2)
red_pixels = np.sum(segmented_8u == 1)
green_pixels = np.sum(segmented_8u == 3)
black_pixels = np.sum(segmented_8u == 0)

blue_pixels_l = np.sum(output_image == 2)
red_pixels_l = np.sum(output_image == 1)
green_pixels_l = np.sum(output_image == 3)
black_pixels_l = np.sum(output_image == 0)

# Print the pixel counts
print(f"Number of blue pixels in segmented image: {blue_pixels}")
print(f"Number of blue pixels in labeled image: {blue_pixels_l}")
print(f"Number of red pixels in segmented image: {red_pixels}")
print(f"Number of red pixels in labeled image: {red_pixels_l}")
print(f"Number of green pixels in segmented image: {green_pixels}")
print(f"Number of green pixels in labeled image: {green_pixels_l}")
print(f"Number of black pixels in segmented image: {black_pixels}")
print(f"Number of black pixels in labeled image: {black_pixels_l}")



from matplotlib import pyplot as plt
plt.imshow(segmented, cmap ='jet')
#plt.imsave('segmented_rock_RF_100_estim.jpg', segmented, cmap ='jet')
plt.title("Segmented Image")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()