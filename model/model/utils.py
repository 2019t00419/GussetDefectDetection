import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            # Convert to grayscale and resize to 28x28
            img = img.convert('L').resize((28, 28))
            images.append(np.array(img))
    return images

def generate_dataset_from_images(root_folder):
    X = []
    Y = []
    
    # Iterate through each class folder
    for class_label in os.listdir(root_folder):
        class_folder = os.path.join(root_folder, class_label)
        
        if os.path.isdir(class_folder):
            images = load_images_from_folder(class_folder)
            # Add images and labels to dataset
            X.extend(images)
            Y.extend([int(class_label)] * len(images))
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y
