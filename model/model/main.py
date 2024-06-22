import os
import numpy as np
from model import gradient_descent, make_predictions, init_params, get_accuracy, test_prediction
from utils import generate_dataset_from_images

def shuffle_data(X):

    p = np.random.permutation(len(X))

    return X[p]

def main():
    # Replace with your folder containing class-separated images
    root_folder = 'F:/UOC/Research/Programs/Test program for edge detection/BalanceOutDetection/model/data/images'

    # Generate dataset from images
    dataset= generate_dataset_from_images(root_folder)

    # Shuffle dataset
    dataset= shuffle_data(dataset)

    # Reshape X_train to (num_features, num_samples)
    no_samples, no_pixels = dataset.shape

    # Split into training and validation sets
    train_ratio = 0.8
    split_index = int(train_ratio * no_samples)


    train_data = (dataset[0:split_index].T)
    X_train_data = train_data[1:no_pixels]/255
    train_labels = train_data[0]

    val_data = (dataset[0:split_index].T)
    X_val_data = val_data[1:no_pixels]/255
    val_labels = val_data[0]

    
    print(f"X_train_data are :  {X_train_data}")
    print(f"train_labels are :  {train_labels}")
    
    print(f"X_val_data are :  {X_val_data}")
    print(f"val_labels are :  {val_labels}")

    

    # Initialize parameters
    W1, b1, W2, b2 = init_params()

    # Train the model
    alpha = 0.10  # Learning rate
    iterations = 500
    W1, b1, W2, b2 = gradient_descent(X_train_data, train_labels, alpha, iterations)


    # Test predictions
    for i in range(4):
        test_prediction(i, W1, b1, W2, b2, X_val_data, val_labels)

if __name__ == "__main__":
    main()
