import cv2 as cv
import numpy as np

def fabric_color(image, second_longest_contour, count, grid_size=30, threshold_value=50):
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Create a mask of the same size as the image, filled with zeros (black)
    mask = np.zeros_like(image)

    # Draw the filled contour on the mask (white)
    cv.drawContours(mask, [second_longest_contour], -1, (255, 255, 255), thickness=cv.FILLED)

    # Mask the original image
    masked_image = cv.bitwise_and(image, mask)

    # Display the results
    cv.imshow('Original Image', image)
    cv.imshow('Masked Image', masked_image)
    cv.imshow('Areas of Significant Deviation', masked_image)
    cv.imwrite(f"images/out/dev/dev_{count}.jpg", masked_image)

def fabric_colos(image, second_longest_contour, count, grid_size=30, threshold_value=50):
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Create a mask of the same size as the image, filled with zeros (black)
    mask = np.zeros_like(image)

    # Draw the filled contour on the mask (white)
    cv.drawContours(mask, [second_longest_contour], -1, (255, 255, 255), thickness=cv.FILLED)

    # Mask the original image
    masked_image = cv.bitwise_and(image, mask)

    # Convert the mask to grayscale and then binary
    mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    _, binary_mask = cv.threshold(mask_gray, 1, 255, cv.THRESH_BINARY)

    # Calculate the overall average color inside the masked area
    mean_val = cv.mean(image, mask=binary_mask)
    overall_average_color = np.array(mean_val[:3], dtype=np.uint8)  # Exclude the alpha channel if present

    # Divide the image into smaller squares and compare their average color with the overall average color
    h, w = image.shape[:2]
    significant_deviation = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            # Define the region of interest (ROI)
            roi = masked_image[y:y + grid_size, x:x + grid_size]
            roi_mask = binary_mask[y:y + grid_size, x:x + grid_size]

            # Calculate the average color inside the ROI
            roi_mean_val = cv.mean(roi, mask=roi_mask)
            roi_average_color = np.array(roi_mean_val[:3], dtype=np.uint8)

            # Calculate the difference from the overall average color
            color_diff = cv.norm(roi_average_color, overall_average_color)

            # If the difference exceeds the threshold, mark the area
            if color_diff > threshold_value:
                significant_deviation[y:y + grid_size, x:x + grid_size] = 255

    # Display the results
    cv.imshow('Original Image', image)
    cv.imshow('Masked Image', masked_image)
    cv.imshow('Areas of Significant Deviation', significant_deviation)
    cv.imwrite(f"images/out/dev/dev_{count}.jpg", significant_deviation)

    