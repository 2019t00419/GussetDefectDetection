import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from miscellaneous import initialize_cam
from contourID import identify_edges
import cv2 as cv

image_width = 64

def shuffle_data(X):

    p = np.random.permutation(len(X))

    return X[p]

def load_images_from_folder(folder, class_label):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            # Convert to grayscale and resize to image_width x image_width
            img = img.convert('L').resize((image_width, image_width))
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

def capture_training_images():
    capture_width, capture_height = 3840, 2160
    cap = initialize_cam(capture_width, capture_height)

    count = 0  # Initialize image count

    while True:
        success, image = cap.read()
        display_image= image.copy()
        if not success:
            break

        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)
        _, cpu_thresholded_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        blurred_otsu = cv.GaussianBlur(cpu_thresholded_image, (5, 5), 0)
        cx = int(capture_width/2)
        cy = int(capture_height/2)

        cv.line(display_image, (int(cx), 0), (int(cx), int(cy)), (0, 255, 0), 2)
        cv.line(display_image, (0,int(cy)), (int(cy), int(cx)), (0, 255, 0), 2)

        cv.imshow("display",display_image)

        # Define the coordinates
        tlx, tly = cx - int(image_width/2), cy - int(image_width/2)  # Top-left corner
        brx, bry = cx + int(image_width/2), cy + int(image_width/2)  # Bottom-right corner

        if abs(tlx - brx) == abs(tly - bry):  # Ensure the coordinates define a square area
            cropped_image = image[tly:bry, tlx:brx]
            grayscale_cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
            #_, otsu_cropped_image = cv.threshold(grayscale_cropped_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

            # Display the cropped image
            cv.imshow("grayscale_cropped_image", grayscale_cropped_image)

            key = cv.waitKey(1)
            if key == ord('0'):  # Check for 'c' key press
                all_folder = 'data/train/front'
                if not os.path.exists(all_folder):
                    os.makedirs(all_folder)
                
                cv.imwrite(os.path.join(all_folder, f"front_{count}.jpg"), grayscale_cropped_image)
                count += 1
                print(f"Image saved: cropped_{count}.jpg")
            if key == ord('1'):  # Check for 'c' key press
                all_folder = 'data/train/back'
                if not os.path.exists(all_folder):
                    os.makedirs(all_folder)
                
                cv.imwrite(os.path.join(all_folder, f"back_{count}.jpg"), grayscale_cropped_image)
                count += 1
                print(f"Image saved: cropped_{count}.jpg")

        if cv.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


#capture_training_images()