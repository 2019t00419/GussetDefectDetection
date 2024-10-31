
from contourID import identify_outer_edge
import cv2 as cv
import numpy as np


import cv2 as cv
import numpy as np

def detect_gusset(contours, display_image, grayscale_image, x_margins, y_margins, frame_width, frame_height, capturedIn, canny, sample_longest_contour, sample_second_longest_contour):
    cx, cy = 0, 0
    MA, ma = 0, 0
    box, longest_contour = None, None
    gusset_detected = False
    captured = capturedIn
    confidence = 0
    
    print("Captured :",captured)
    # Early return if contours is empty
    if not contours:
        print("No contours available, returning default values.")
        #cv.imshow("display image",display_image)
        return gusset_detected, cx, cy, box, longest_contour, display_image, grayscale_image, captured, ma, MA, confidence
    
    # Identify the outer edge of the gusset
    longest_contour = identify_outer_edge(contours, sample_longest_contour)

    if longest_contour is not None:
        (x, y), (MA, ma), angle = cv.fitEllipse(longest_contour)

        # Calculate confidence using matchShapes
        ret = cv.matchShapes(longest_contour, sample_longest_contour, 1, 0.0)
        confidence = max((1 - ret) * 100, 0)  # Ensure confidence is at least 0

        if confidence > 80:
            x, y, w, h = cv.boundingRect(longest_contour)
            cv.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if contour lies within the margins
            if (x < x_margins) or (y < y_margins) or ((x + w) > (frame_width - x_margins)) or ((y + h) > (frame_height - y_margins)):
                captured = False
            else:
                # Draw the gusset's bounding box
                rect = cv.minAreaRect(longest_contour)
                box = cv.boxPoints(rect)
                box = np.int0(box)

                # Calculate the center of the bounding box
                M = cv.moments(box)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv.circle(display_image, (cx, cy), 5, (255, 0, 0), cv.FILLED)
                    gusset_detected = True
                else:
                    print("Moment calculation failed; m00 is zero.")
    
    else:
        buffer_variable="Longest contour is not available."
    
    # Return values with guaranteed structure
    return gusset_detected, cx, cy, box, longest_contour, display_image, grayscale_image, captured, ma, MA, confidence
