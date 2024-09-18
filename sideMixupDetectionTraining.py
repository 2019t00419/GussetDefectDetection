import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import os
import seaborn as sns
import pickle
from sideMixupDetectionFilters import resizer, feature_extractor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import confusion_matrix
from sklearn import metrics

size = 64

# Resize images to
# Capture images and labels into arrays.
# Start by creating empty lists.
train_images = []
train_labels = [] 
for directory_path in glob.glob("test/SMD/Train_images/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        imgc = cv.imread(img_path, cv.IMREAD_COLOR)  # Reading color images
        
        imgc = resizer(imgc, size)
        img = cv.cvtColor(imgc, cv.COLOR_RGB2GRAY)  # Optional step. Change BGR to RGB
        train_images.append(img)
        train_labels.append(label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Do exactly the same for test/validation images
test_images = []
test_labels = [] 
for directory_path in glob.glob("test/SMD/validation/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        imgc = cv.imread(img_path, cv.IMREAD_COLOR)
        img = cv.cvtColor(imgc, cv.COLOR_BGR2GRAY)  # Optional.
        test_images.append(img)
        test_labels.append(fruit_label)
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Label encoding
le = preprocessing.LabelEncoder()
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
test_labels_encoded = le.transform(test_labels)

# Ensure labels are 1D using .ravel()
y_train = train_labels_encoded.ravel()
y_test = test_labels_encoded.ravel()

# Normalize pixel values to between 0 and 1
x_train, x_test = train_images / 255.0, test_images / 255.0

####################################################################
# Feature extraction
image_features = feature_extractor(x_train, imgc)

# Reshape to a vector for Random Forest / SVM training
X_for_RF = np.reshape(image_features, (x_train.shape[0], -1))  # Reshape to #images, features

# Train Random Forest
RF_model = RandomForestClassifier(n_estimators=50, random_state=42)
RF_model.fit(X_for_RF, y_train)

# Save the trained model and label encoder
pickle.dump(RF_model, open("sideMixupdetectionModel", 'wb'))
pickle.dump(le, open("SMD_encoder.pkl", 'wb'))

# Predict on Test data
test_features = feature_extractor(x_test, imgc)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

# Predict on test data
test_prediction = RF_model.predict(test_for_RF)
test_prediction = le.inverse_transform(test_prediction)

# Print overall accuracy
print("Accuracy =", metrics.accuracy_score(test_labels, test_prediction))

# Confusion matrix
cm = confusion_matrix(test_labels, test_prediction)
fig, ax = plt.subplots(figsize=(6, 6))  # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, ax=ax)

# Check results on a few random images
n = random.randint(0, x_test.shape[0] - 1)  # Select random index for testing
img = x_test[n]
#cv.imshow("img", img)

# Predict for a single image from the test set
input_img = np.expand_dims(img, axis=0)
input_img_features = feature_extractor(input_img, imgc)
input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))

img_prediction = RF_model.predict(input_img_for_RF)
img_prediction = le.inverse_transform(img_prediction.ravel())  # Ensure the label is correctly decoded
print("The prediction for this image is:", img_prediction)
print("The actual label for this image is:", test_labels[n])
