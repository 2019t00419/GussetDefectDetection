import numpy as np
import cv2 as cv
import pandas as pd
import pickle
from detectionAssistfilters import detection_filters
import os
from sklearn import metrics
import time
import threading
import sys

colour = "Skin"  # Options: Bianco, Skin, Nero

# Start total execution timer
total_start = time.time()
resize_factor = 4.5
image_dataset = pd.DataFrame()  # DataFrame to capture image features

# Function to display a spinner animation
def spinner():
    while training:
        for char in "|/-\\":
            sys.stdout.write(f'\rTraining model... {char}')
            sys.stdout.flush()
            time.sleep(0.2)

# SECTION 1: Load and preprocess images
img_path = "test/Train_images/"
image_start = time.time()  # Timer for image loading section
for image in os.listdir(img_path):
    print(image)
    
    df = pd.DataFrame()  # Temporary data frame to capture information for each loop.
    
    input_img = cv.imread(img_path + image)  # Read images
    detection_height = 2160  # Detection width is set for Landscape images

    input_image_width, input_image_height, _ = input_img.shape
    if input_image_width < input_image_height:  # Portrait
        input_img = cv.rotate(input_img, cv.ROTATE_90_CLOCKWISE)
    
    input_image_width, input_image_height, _ = input_img.shape
    resize_factor = input_image_height / detection_height
    input_img = cv.resize(input_img, (int(input_image_height / resize_factor), int((input_image_width / input_image_height) * (input_image_height / resize_factor))))
    
    if input_img.ndim == 3 and input_img.shape[-1] == 3:   
        if colour == "Bianco" or colour == "Skin":
            img = input_img[:, :, 2]  # Red channel
        elif colour == "Nero":
            img = input_img[:, :, 1]  # Green channel
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("The module works only with grayscale and RGB images!")
    pixel_values = img.reshape(-1)
    df['Original Image'] = pixel_values
    df['Image_Name'] = image
    
    df = detection_filters(img, df, input_img)
    image_dataset = pd.concat([image_dataset, df], ignore_index=True)

image_end = time.time()  # End timer for image loading section
print(f"Time taken for loading and preprocessing images: {image_end - image_start} seconds")

# SECTION 2: Load and preprocess masks
mask_dataset = pd.DataFrame()
mask_start = time.time()  # Timer for mask loading section
mask_path = "test/Train_lables/"
for mask in os.listdir(mask_path):
    print(mask)
    
    df2 = pd.DataFrame()
    input_mask = cv.imread(mask_path + mask)
    
    input_image_width, input_image_height, _ = input_mask.shape
    if input_image_width < input_image_height:
        input_mask = cv.rotate(input_mask, cv.ROTATE_90_CLOCKWISE)
    
    resize_factor = input_image_height / detection_height
    input_mask = cv.resize(input_mask, (int(input_image_height / resize_factor), int((input_image_width / input_image_height) * (input_image_height / resize_factor))))
    
    output_image = np.zeros((input_mask.shape[0], input_mask.shape[1]), dtype=np.uint8)
    blue_channel, green_channel, red_channel = input_mask[:, :, 0], input_mask[:, :, 1], input_mask[:, :, 2]

    output_image[(red_channel > 0) & (blue_channel == 0) & (green_channel == 0)] = 1
    output_image[(green_channel > 0) & (blue_channel == 0) & (red_channel == 0)] = 2
    output_image[(blue_channel > 0) & (green_channel == 0) & (red_channel == 0)] = 3
    output_image[(blue_channel > 0) & (green_channel == 0) & (red_channel > 0)] = 4

    label_values = output_image.reshape(-1)
    df2['Label_Value'] = label_values
    df2['Mask_Name'] = mask
    
    mask_dataset = pd.concat([mask_dataset, df2], ignore_index=True)

mask_end = time.time()  # End timer for mask loading section
print(f"Time taken for loading and preprocessing masks: {mask_end - mask_start} seconds")

# SECTION 3: Combine datasets and train/test split
combine_start = time.time()
dataset = pd.concat([image_dataset, mask_dataset], axis=1)

if dataset['Image_Name'].equals(dataset['Mask_Name']):
    print("Train and Mask file names match")
    dataset = dataset[dataset.Label_Value != 0]
    X = dataset.drop(labels=["Image_Name", "Mask_Name", "Label_Value"], axis=1)
    Y = dataset["Label_Value"].values

    from sklearn.preprocessing import LabelEncoder
    Y = LabelEncoder().fit_transform(Y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
else:
    print("Train and Mask file names do not match")

combine_end = time.time()
print(f"Time taken for combining datasets and train/test split: {combine_end - combine_start} seconds")

# SECTION 4: Train the model with spinner animation
from sklearn.ensemble import RandomForestClassifier

training = True  # Flag to indicate training status
train_start = time.time()

# Start the spinner in a separate thread
spinner_thread = threading.Thread(target=spinner)
spinner_thread.start()

# Model training
model = RandomForestClassifier(n_estimators=25, random_state=42) # Adjust `n_jobs` as needed
model.fit(X_train, y_train)

# Stop the spinner
training = False
spinner_thread.join()

train_end = time.time()
print(f"\nTime taken for training the model: {train_end - train_start} seconds")

# SECTION 5: Model evaluation with Mean IoU
evaluate_start = time.time()
prediction_test = model.predict(X_test)
print("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

# Mean IoU Calculation
def mean_iou(y_true, y_pred, num_classes):
    iou_list = []
    for c in range(num_classes):
        true_class = (y_true == c)
        pred_class = (y_pred == c)
        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()
        if union == 0:
            iou = 1
        else:
            iou = intersection / union
        iou_list.append(iou)
    return np.mean(iou_list)

mean_iou_score = mean_iou(y_test, prediction_test, num_classes=4)  # Adjust number of classes as needed
print("Mean IoU:", mean_iou_score)

from yellowbrick.classifier import ROCAUC
roc_auc = ROCAUC(model, classes=[0, 1, 2, 3, 4])
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)
roc_auc.show()
evaluate_end = time.time()
print(f"Time taken for model evaluation: {evaluate_end - evaluate_start} seconds")

# SECTION 6: Save the model
save_start = time.time()
model_name = "detectionSupportModelHighRes"
pickle.dump(model, open(model_name, 'wb'))
save_end = time.time()
print(f"Time taken for saving the model: {save_end - save_start} seconds")

# Total execution time
total_end = time.time()
print(f"Total execution time: {total_end - total_start} seconds")
# Print feature importance


# Calculate and display feature importance
feature_importance = model.feature_importances_
feature_names = X.columns  # Get the names of the features from the DataFrame columns

# Create a DataFrame for easy visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Display top 10 features only
top_10_importance_df = importance_df.head(10)

print("\nTop 10 Feature Importance:")
print(top_10_importance_df)

# Optional: Plot the feature importance for a more visual representation
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(top_10_importance_df['Feature'], top_10_importance_df['Importance'], color='skyblue')
plt.xlabel("Importance")
plt.title("Top 10 Feature Importance from Random Forest")
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()
