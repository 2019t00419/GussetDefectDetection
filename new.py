import numpy as np
import cv2 as cv

def new_feature(original_frame, longest_contour, count):
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
    cv.imshow("Otsu cropped Image", otsu_cropped_image)
    cv.imwrite("images/out/cropped/cropped (" + str(count) + ").jpg", otsu_cropped_image)

# Example usage:
# original_frame = cv.imread('path_to_image.jpg')
# longest_contour = some_contour_extracted_earlier
# count = 1
# new_feature(original_frame, longest_contour, count)
