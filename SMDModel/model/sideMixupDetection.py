import os
import numpy as np
from model import gradient_descent, make_predictions, init_params, get_accuracy, test_prediction
from SMDUtils import generate_dataset_from_images

def shuffle_data(X):

    p = np.random.permutation(len(X))

    return X[p]

def detectSide(mode):
    if(mode=="train"):
        # Replace with your folder containing class-separated images
        train_folder = 'model/data/images/train'

        # Generate dataset from images
        train_dataset= generate_dataset_from_images(train_folder)

        # Shuffle dataset
        train_dataset= shuffle_data(train_dataset)

        # Reshape X_train to (num_features, num_samples)
        no_samples, no_pixels = train_dataset.shape

        #train_data = (dataset[0:split_index].T)
        train_data = train_dataset.T
        X_train_data = train_data[1:no_pixels]/255
        train_labels = train_data[0]

        
        print(f"X_train_data are :  {X_train_data}")
        print(f"train_labels are :  {train_labels}")

        

        # Initialize parameters
        W1, b1, W2, b2 = init_params()

        # Train the model
        alpha = 0.10  # Learning rate
        iterations = 500
        W1, b1, W2, b2 = gradient_descent(X_train_data, train_labels, alpha, iterations)
        np.save('model/W1.npy', W1)
        np.save('model/b1.npy', b1)
        np.save('model/W2.npy', W2)
        np.save('model/b2.npy', b2)

    elif(mode=="check"):

        
        valid_folder = 'model/data/images/valid'
        
        valid_dataset= generate_dataset_from_images(valid_folder)
        
        valid_dataset= shuffle_data(valid_dataset)

        
        no_samples, no_pixels = valid_dataset.shape

        

        val_data = (valid_dataset.T)
        X_val_data = val_data[1:no_pixels]/255
        val_labels = val_data[0]

        
        
        print(f"X_val_data are :  {X_val_data}")
        print(f"val_labels are :  {val_labels}")

        # Test predictions
        W1 = np.load('model/W1.npy', allow_pickle=True)
        b1 = np.load('model/b1.npy', allow_pickle=True)
        W2 = np.load('model/W2.npy', allow_pickle=True)
        b2 = np.load('model/b2.npy', allow_pickle=True)

        for i in range(len(val_labels)):
            test_prediction(i, W1, b1, W2, b2, X_val_data, val_labels)


if __name__ == "__main__":
    detectSide(mode="check")
