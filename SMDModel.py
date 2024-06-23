import numpy as np
from matplotlib import pyplot as plt
from SMDUtils import generate_dataset_from_images,shuffle_data
import cv2 as cv

image_width = 100
no_of_neurons = 10
classes = ["Front","Back"]

def init_params():
    W1 = np.random.rand(no_of_neurons, (image_width**2)) - 0.5
    b1 = np.random.rand(no_of_neurons, 1) - 0.5
    W2 = np.random.rand(len(classes), no_of_neurons) - 0.5
    b2 = np.random.rand(len(classes), 1) - 0.5
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    
    # Initialize lists to store iteration number and accuracy
    iteration_list = []
    accuracy_list = []
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(accuracy)
            
            # Append iteration number and accuracy to lists
            iteration_list.append(i)
            accuracy_list.append(accuracy)
            
            # Plot the accuracy
            plt.plot(iteration_list, accuracy_list, label="Accuracy")
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Iterations')
            plt.grid(True)
            plt.pause(0.05)  # Pause to update the plot
            
    plt.show()
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def test_prediction(index, W1, b1, W2, b2, X_train, Y_train):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print(f"Prediction: {classes[prediction[0]]}")
    print(f"Label: {classes[label]}")
    
    current_image = current_image.reshape((100, 100)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    return prediction

def detect_side(image_cropped):
    # Resize to 100x100 pixels
    img_resized = cv.resize(image_cropped, (100, 100))
    # Flatten the image to a 1D array
    flattened_img = img_resized.flatten()
    # Normalize the pixel values
    normalized_img = flattened_img / 255.0
    # Reshape to (1, 10000)
    reshaped_img = normalized_img.reshape(1, -1)

    print(f"Flattened image shape: {flattened_img.shape}")
    print(f"Normalized image shape (after reshape): {reshaped_img.shape}")


    W1 = np.load('SMDModel/W1.npy', allow_pickle=True)
    b1 = np.load('SMDModel/b1.npy', allow_pickle=True)
    W2 = np.load('SMDModel/W2.npy', allow_pickle=True)
    b2 = np.load('SMDModel/b2.npy', allow_pickle=True)

    # Make prediction on the processed image
    prediction = make_predictions(reshaped_img.T, W1, b1, W2, b2)
    
    print(f"Prediction: {classes[prediction[0]]}")
    
    reverted_image = reshaped_img.reshape((100, 100)) * 255
    plt.gray()
    plt.imshow(reverted_image, interpolation='nearest')
    plt.show()

    return prediction


def detect_side_image_source(image_path):

    
    img = cv.imread(image_path)
    if img is None:
        print("Error: Image not found or invalid path.")
        return None

    # Convert to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Resize to 100x100 pixels
    img_resized = cv.resize(img_gray, (100, 100))
    # Flatten the image to a 1D array
    flattened_img = img_resized.flatten()
    # Normalize the pixel values
    normalized_img = flattened_img / 255.0
    # Reshape to (1, 10000)
    reshaped_img = normalized_img.reshape(1, -1)

    print(f"Flattened image shape: {flattened_img.shape}")
    print(f"Normalized image shape (after reshape): {reshaped_img.shape}")


    W1 = np.load('SMDModel/W1.npy', allow_pickle=True)
    b1 = np.load('SMDModel/b1.npy', allow_pickle=True)
    W2 = np.load('SMDModel/W2.npy', allow_pickle=True)
    b2 = np.load('SMDModel/b2.npy', allow_pickle=True)

    # Make prediction on the processed image
    prediction = make_predictions(reshaped_img.T, W1, b1, W2, b2)
    
    print(f"Prediction: {classes[prediction[0]]}")
    
    reverted_image = reshaped_img.reshape((100, 100)) * 255
    plt.gray()
    plt.imshow(reverted_image, interpolation='nearest')
    plt.show()

    return prediction



def crop_image(original_frame, longest_contour, count):
    M = cv.moments(longest_contour)

    frame_height, frame_width, channels = original_frame.shape
    cx = int(frame_width * (M['m10'] / M['m00']) / 960)
    cy = int(frame_height * (M['m01'] / M['m00']) / 1280)

    print("Center point = (" + str(cx) + "," + str(cy) + ")")

    # Define the coordinates
    tlx, tly = cx - 50, cy - 50  # Top-left corner
    brx, bry = cx + 50, cy + 50  # Bottom-right corner

    print("Top left point = (" + str(tlx) + "," + str(tly) + ")")
    print("Bottom right point = (" + str(brx) + "," + str(bry) + ")")

    # Ensure the coordinates define a square area
    if abs(tlx - brx) != abs(tly - bry):
        raise ValueError("The provided coordinates do not define a square area.")

    # Crop the image
    cropped_image = original_frame[tly:bry, tlx:brx]
    grayscale_cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
    _, otsu_cropped_image = cv.threshold(grayscale_cropped_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Display the cropped image
    #cv.imshow("Otsu cropped Image", otsu_cropped_image)
    cv.imwrite("images/out/cropped/cropped (" + str(count) + ").jpg", otsu_cropped_image)
    detect_side(otsu_cropped_image)

