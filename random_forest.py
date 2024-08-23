
import numpy as np
import cv2
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier



#import the training mask
#training mask should be labled using colours
#black = background , red = feature 0, green = feature 1, blue = feature 2
labeled_img = cv2.imread('test/Train_lables/hh0.jpg', cv2.IMREAD_COLOR)

#import the training image as colored image
imgColor= cv2.imread('test/Train_images/h1.jpg')
#convert training image to grayscale
img = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)  
#show training image
cv2.imshow("original image", img)
#applying gaussian blur to reduce the details on fabric. Optional. May be removed when using multiple images to train
img = cv2.GaussianBlur(img, (9, 9), 0)
#reshape  the training image to a column 
img2 = img.reshape(-1)
#create a dataframe to store pixel data
df = pd.DataFrame()
#append the pixel data from the training image under the label Original Image 
df['Original Image'] = img2
#print(df)


#Generating multiple gabor features
num = 1
kernels = []

#number of thetas
for theta in range(18,24):
    #calculating theta for each iteration. max 2pi
    theta = theta / 16. * np.pi
    #sigma values list
    for sigma in (1, 3, 5, 7, 9):
        #range of wavelengths. 0 to pi with increments of pi/4
        for lamda in np.arange(0, np.pi, np.pi / 4):
            #gamma values list
            for gamma in (0.05, 0.1, 0.5):     
                #label for the gabor kernel  
                gabor_label = 'Gabor' + str(num)
                #print(gabor_label)
                #kernal size
                ksize=30
                #generating gabor kernal with the specific values
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                #append the new kernal to the kernals
                kernels.append(kernel)
                #apply the gabor kernal to the training image. fmg = filtered 2D image 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                #reshape the 2d filtered image to a column
                filtered_img = fimg.reshape(-1)
                #append the filtered image pixels to the data frame under the gabor kernal name
                df[gabor_label] = filtered_img
                #print parameters used in the gabor kernal
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1
                
#generate other filters for better feature extraction
         
print(df)       
#apply canny
edges = cv2.Canny(img, 100,200)
#reshape
edges1 = edges.reshape(-1)
#add to the dataframe
df['Canny Edge'] = edges1


#apply roberts edge
edge_roberts = roberts(img)
edge_roberts1 = edge_roberts.reshape(-1)
df['Roberts'] = edge_roberts1

#apply sobel
edge_sobel = sobel(img)
edge_sobel1 = edge_sobel.reshape(-1)
df['Sobel'] = edge_sobel1

#apply scharr
edge_scharr = scharr(img)
edge_scharr1 = edge_scharr.reshape(-1)
df['Scharr'] = edge_scharr1

#apply prewitt
edge_prewitt = prewitt(img)
edge_prewitt1 = edge_prewitt.reshape(-1)
df['Prewitt'] = edge_prewitt1

#apply gaussian with sigma=3
from scipy import ndimage as nd
gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian_img1 = gaussian_img.reshape(-1)
df['Gaussian s3'] = gaussian_img1

#apply gaussian with sigma=7
gaussian_img2 = nd.gaussian_filter(img, sigma=7)
gaussian_img3 = gaussian_img2.reshape(-1)
df['Gaussian s7'] = gaussian_img3

#apply median with sigma=3
median_img = nd.median_filter(img, size=3)
median_img1 = median_img.reshape(-1)
df['Median s3'] = median_img1


# Calculate entropy
#entr_img = entropy(img, disk(10))
#Normalize entropy image to the range [0, 1]
#entr_img = (entr_img - entr_img.min()) / (entr_img.max() - entr_img.min())
#print (entr_img.shape)
#entr_img1 = entr_img.reshape(-1)
#df['entropy'] = entr_img1  #Add column to original dataframe

#using the labled image loaded at  the begining
#print(labeled_img.shape)
#print("labeled_img")
#print(labeled_img)
cv2.imshow("labeled_img", labeled_img)


#create an empty numpy array for the output image with the same dimensions as the input
output_image = np.zeros((labeled_img.shape[0], labeled_img.shape[1]), dtype=np.uint8)

#extract the colour channels of the labeled image
blue_channel = labeled_img[:, :, 0]
green_channel = labeled_img[:, :, 1]
red_channel = labeled_img[:, :, 2]

#identify segments using the colour and assign gray scale pixel values
output_image[(red_channel > 0) & (blue_channel == 0) & (green_channel == 0)] = 1
output_image[(green_channel > 0) & (blue_channel == 0) & (red_channel == 0)] = 2
output_image[(blue_channel > 0) & (green_channel == 0) & (red_channel == 0)] = 3

#reshape the new labled image after remapping grayscale values
labeled_img1 = output_image.reshape(-1)
#append the labled image to the dataframe
df['Labels'] = labeled_img1
#print(df)

#define an image to test in the prediction stage later by dropping the labels
original_img_data = df.drop(labels = ["Labels"], axis=1)
#print(original_img_data)
#df.to_csv("Gabor.csv")

#drop the labels with value 0 as they are not important for training the model
df = df[df.Labels != 0]
#print(df)

#define the dependent variable that needs to be predicted
Y = df["Labels"].values

#encode Y values to 0, 1, 2, 3, ....
Y = LabelEncoder().fit_transform(Y)

#print(Y)

#define the independent variables
X = df.drop(labels = ["Labels"], axis=1) 
#print(X)

#split pixels into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

#load the random forest classifier
#instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 20, random_state = 42)

#train the model on training data
model.fit(X_train, y_train)

#print the importance of each generated features
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)

prediction_RF = model.predict(X_test)
from sklearn import metrics
#Print the prediction accuracy
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy using Random Forest= ", metrics.accuracy_score(y_test, prediction_RF)*100,"%")

from yellowbrick.classifier import ROCAUC

print("Classes in the image are: ", np.unique(Y))

#ROC curve for RF
roc_auc=ROCAUC(model, classes=[0, 1, 2, 3])  #Create object
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)


import pickle

#Save the trained model as pickle string to disk for future use
filename = "fabric_defect_model"
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
segmented_bgr_image[segmented_8u == 0] = [0, 0, 255]  # Red
segmented_bgr_image[segmented_8u == 1] = [0, 0, 0]  # Green
segmented_bgr_image[segmented_8u == 2] = [255, 0, 0]  # Blue

cv2.imshow("segmented_bgr_image", segmented_bgr_image)



blue_pixels = np.sum(segmented_8u == 3)
green_pixels = np.sum(segmented_8u == 2)
red_pixels = np.sum(segmented_8u == 1)
black_pixels = np.sum(segmented_8u == 0)

blue_pixels_l = np.sum(output_image == 3)
green_pixels_l = np.sum(output_image == 2)
red_pixels_l = np.sum(output_image == 1)
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


roc_auc.show()
  

from matplotlib import pyplot as plt
plt.imshow(segmented, cmap ='jet')
#plt.imsave('segmented_rock_RF_100_estim.jpg', segmented, cmap ='jet')
plt.title("Segmented Image")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()