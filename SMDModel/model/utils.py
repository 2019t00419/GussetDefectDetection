import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder, class_label):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            # Convert to grayscale and resize to 100 x 100
            img = img.convert('L').resize((100, 100))
            # Flatten the image to 1D array
            flattened_img = np.array(img).flatten()
            # Prepend the class label to the flattened image array
            labeled_img = np.insert(flattened_img, 0, class_label)
            images.append(labeled_img)
    return images

def generate_dataset_from_images(root_folder):
    X = []
    
    # Iterate through each class folder
    for class_label in os.listdir(root_folder):
        class_folder = os.path.join(root_folder, class_label)
        
        if os.path.isdir(class_folder):
            images = load_images_from_folder(class_folder, class_label)
            # Add images and labels to dataset
            X.extend(images)
            print(f"Class label : {class_label}")
    
    X = np.array(X)
    
    return X
