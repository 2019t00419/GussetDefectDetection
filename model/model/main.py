import os
import numpy as np
from model import gradient_descent, make_predictions, init_params, get_accuracy, test_prediction
from utils import generate_dataset_from_images

def shuffle_data(X, Y):
    """
    Shuffle X and Y arrays in unison.
    """
    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    return X[p], Y[p]

def main():
    # Replace with your folder containing class-separated images
    root_folder = 'F:/UOC/Research/Programs/Test program for edge detection/BalanceOutDetection/model/data/images'

    # Generate dataset from images
    X_train, Y_train = generate_dataset_from_images(root_folder)

    # Shuffle dataset
    X_train, Y_train = shuffle_data(X_train, Y_train)

    # Normalize pixel values
    X_train = X_train / 255.

    # Reshape X_train to (num_features, num_samples)
    num_samples, img_height, img_width = X_train.shape
    X_train = X_train.reshape(num_samples, img_height * img_width).T

    # Split into training and validation sets
    train_ratio = 0.8
    split_index = int(train_ratio * len(X_train))

    X_train_data, X_val_data = X_train[:, :split_index], X_train[:, split_index:]
    Y_train_data, Y_val_data = Y_train[:split_index], Y_train[split_index:]

    # Initialize parameters
    W1, b1, W2, b2 = init_params()

    # Train the model
    alpha = 0.10  # Learning rate
    iterations = 500
    W1, b1, W2, b2 = gradient_descent(X_train_data, Y_train_data, alpha, iterations)

    # Test predictions
    for i in range(4):
        test_prediction(i, W1, b1, W2, b2, X_train_data, Y_train_data)

if __name__ == "__main__":
    main()
