
import numpy as np
import cv2 as cv
import pandas as pd
import pickle
from filter_generation import filter_generation
import os



image_dataset = pd.DataFrame()  #Dataframe to capture image features

img_path = "test/Train_images/"
for image in os.listdir(img_path):  #iterate through each file 
    print(image)
    
    df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
    #Reset dataframe to blank after each loop.
    
    input_img = cv.imread(img_path + image)  #Read images
    
    #Check if the input image is RGB or grey and convert to grey if RGB
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv.cvtColor(input_img,cv.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("The module works only with grayscale and RGB images!")
        
        
    #img = cv.GaussianBlur(img, (9, 9), 0)
    #Add pixel values to the data frame
    pixel_values = img.reshape(-1)
    df['Original Image'] = pixel_values   #Pixel value itself as a feature
    df['Image_Name'] = image   #Capture image name as we read multiple images
    
    print (df)

    df = filter_generation(img,df)

    
######################################                    
#Update dataframe for images to include details for each image in the loop
    image_dataset = pd.concat([image_dataset, df], ignore_index=True)


###########################################################
# STEP 2: READ LABELED IMAGES (MASKS) AND CREATE ANOTHER DATAFRAME
    # WITH LABEL VALUES AND LABEL FILE NAMES
##########################################################
mask_dataset = pd.DataFrame()  #Create dataframe to capture mask info.

mask_path = "test/Train_lables/"    
for mask in os.listdir(mask_path):  #iterate through each file to perform some action
    print(mask)
    
    df2 = pd.DataFrame()  #Temporary dataframe to capture info for each mask in the loop
    input_mask = cv.imread(mask_path + mask)
   
    #create an empty numpy array for the output image with the same dimensions as the input
    output_image = np.zeros((input_mask.shape[0], input_mask.shape[1]), dtype=np.uint8)

    #extract the colour channels of the labeled image
    blue_channel = input_mask[:, :, 0]
    green_channel = input_mask[:, :, 1]
    red_channel = input_mask[:, :, 2]

    #identify segments using the colour and assign gray scale pixel values
    output_image[(red_channel > 0) & (blue_channel == 0) & (green_channel == 0)] = 1
    output_image[(green_channel > 0) & (blue_channel == 0) & (red_channel == 0)] = 2
    output_image[(blue_channel > 0) & (green_channel == 0) & (red_channel == 0)] = 3

    #output_image = cv.cvtColor(output_image,cv.COLOR_BGR2GRAY)
    #Add pixel values to the data frame
    label_values = output_image.reshape(-1)
    df2['Label_Value'] = label_values
    df2['Mask_Name'] = mask
    
    mask_dataset = pd.concat([mask_dataset, df2], ignore_index=True)
  #Update mask dataframe with all the info from each mask

################################################################
 #  STEP 3: GET DATA READY FOR RANDOM FOREST (or other classifier)
    # COMBINE BOTH DATAFRAMES INTO A SINGLE DATASET
###############################################################
dataset = pd.concat([image_dataset, mask_dataset], axis=1)    #Concatenate both image and mask datasets
print(dataset.shape)
#If you expect image and mask names to be the same this is where we can perform sanity check
#dataset['Image_Name'].equals(dataset['Mask_Name'])   
##
##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0.
dataset = dataset[dataset.Label_Value != 0]

#Assign training features to X and labels to Y
#Drop columns that are not relevant for training (non-features)
X = dataset.drop(labels = ["Image_Name", "Mask_Name", "Label_Value"], axis=1) 

#Assign label values to Y (our prediction)
Y = dataset["Label_Value"].values 

#Encode Y values to 0, 1, 2, 3, .... (NOt necessary but makes it easy to use other tools like ROC plots)
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)


##Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

####################################################################
# STEP 4: Define the classifier and fit a model with our training data
###################################################################

#Import training classifier
from sklearn.ensemble import RandomForestClassifier
## Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

## Train the model on training data
model.fit(X_train, y_train)

#print the importance of each generated features
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)

#######################################################
# STEP 5: Accuracy check
#########################################################

from sklearn import metrics
prediction_test = model.predict(X_test)
##Check accuracy on test dataset. 
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

from yellowbrick.classifier import ROCAUC
print("Classes in the image are: ", np.unique(Y))

#ROC curve for RF
roc_auc=ROCAUC(model, classes=[0, 1, 2, 3])  #Create object
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)
roc_auc.show()

##########################################################
#STEP 6: SAVE MODEL FOR FUTURE USE
###########################################################
##You can store the model for future use. In fact, this is how you do machine elarning
##Train on training images, validate on test images and deploy the model on unknown images. 
#
#
##Save the trained model as pickle string to disk for future use
model_name = "fabric_defect_model_multi_image"
pickle.dump(model, open(model_name, 'wb'))
#
##To test the model on future datasets
#loaded_model = pickle.load(open(model_name, 'rb'))


