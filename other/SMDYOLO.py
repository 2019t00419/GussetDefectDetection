from ultralytics import YOLO
import cv2 as cv

image_width = 64

def train_model():
    model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
    results = model.train(data="F:/UOC/Research/Programs/Test program for edge detection/BalanceOutDetection/data", epochs=20, imgsz=64)

def infer_image(cropped_image):
    # Load the trained model
    model = YOLO("runs/classify/train5/weights/best.pt")  # load your custom model

    # Perform inference on the image
    results = model(cropped_image)  # predict on an image

    # Extract class predictions and confidence scores
    for result in results:
        if hasattr(result, 'probs'):
            # Print top-1 class and confidence
            label_top1 = result.names[result.probs.top1]
            confidence_top1 = result.probs.top1conf.item()
            print(f"Side: {label_top1}, Confidence: {confidence_top1:.2f}")
        else:
            print("The results object does not have a 'probs' attribute. Here's the full results object:")
            print(result)
    return label_top1



def crop_image(original_frame, longest_contour, count):
    M = cv.moments(longest_contour)
    if M['m00'] == 0:
        return None 
    else:
        frame_height, frame_width, channels = original_frame.shape
        print(original_frame.shape)
        cx = int(frame_width * (M['m10'] / M['m00']) / 960)
        cy = int(frame_height * (M['m01'] / M['m00']) / 1280)

        print("Center point = (" + str(cx) + "," + str(cy) + ")")

        # Define the coordinates
        tlx, tly = cx - int(image_width/2), cy - int(image_width/2)  # Top-left corner
        brx, bry = cx + int(image_width/2), cy + int(image_width/2)  # Bottom-right corner

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
        ##cv.imshow("Otsu cropped Image", otsu_cropped_image)
        cv.imwrite("images/out/cropped/cropped (" + str(count) + ").jpg", grayscale_cropped_image)
        #fabric_side = detect_side(otsu_cropped_image)
        fabric_side = infer_image(grayscale_cropped_image)
    return fabric_side

