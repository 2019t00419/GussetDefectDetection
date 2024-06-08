import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture(0)  # Use the webcam

cpu_times = []
last_update_time = time.time()
update_interval = 1  # Update FPS every second
avg_cpu_fps = 0  # Initialize average CPU FPS
captured = False
detection_length = 600
detection_height = 460

count = 0

def detectGusset():
    global captured
    global count
    global cpu_times
    global avg_cpu_fps
    global last_update_time

    frame_start_time = time.time()
    
    start_cpu = time.time()
    start_open = time.time()
    success, image = cap.read()
    display_image = image.copy() if success else None
    if not success:
        print("Failed to load video")
        return None

    end_open = time.time()
    open_time = (end_open - start_open) * 1000
    print("Open time : " + str(open_time) + "ms")
    
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    detection_mask = np.zeros_like(grayscale_image)

    frame_height, frame_width = grayscale_image.shape
    x_margins = int((frame_width - detection_length) / 2)
    y_margins = int((frame_height - detection_height) / 2)

    # Create a rectangular mask
    cv.rectangle(detection_mask, (x_margins, y_margins), (frame_width - x_margins, frame_height - y_margins), 255, cv.FILLED)

    # Draw the rectangle on the display image for visualization
    cv.rectangle(display_image, (x_margins, y_margins), (frame_width - x_margins, frame_height - y_margins), (255, 255, 255), 2)

    # Apply the mask to the grayscale image
    masked_grayscale_image = cv.bitwise_and(grayscale_image, grayscale_image, mask=detection_mask)

    # Show the masked grayscale image
    # CPU operations
    blurred_image = cv.GaussianBlur(masked_grayscale_image, (5, 5), 0)
    _, cpu_thresholded_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(cpu_thresholded_image, (5, 5), 0)
    canny = cv.Canny(blurred_otsu, 100, 200)
    
    # Find contours and draw the bounding box of the largest contour
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)

        x, y, w, h = cv.boundingRect(largest_contour)
        cv.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if (x < x_margins) or (y < y_margins) or ((x+w) > (frame_width-x_margins)) or ((y+h) > (frame_height-y_margins)):
            captured = False
        else:
            rect = cv.minAreaRect(largest_contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            M = cv.moments(box)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv.circle(display_image, (cx, cy), 5, (255, 0, 0), cv.FILLED)

            if cx > (frame_width / 2) and not captured:
                captured = True
                captureFrame(count)
                count += 1

            cv.drawContours(display_image, [box], 0, (0, 0, 255), 2)

            (x, y), (MA, ma), angle = cv.fitEllipse(largest_contour)
            cv.putText(display_image, f"Major axis length: {int(MA)}    Minor axis length: {int(ma)}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            
            # Create a mask for the largest contour
            mask = np.zeros_like(grayscale_image)
            cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)
            cv.drawContours(canny, [largest_contour], -1, 0, 2)

            # Mask the Canny image
            masked_canny = cv.bitwise_and(canny, canny, mask=mask)

            # Find contours inside the masked canny image
            inner_contours, _ = cv.findContours(masked_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if inner_contours:
                largest_inner_contour = max(inner_contours, key=cv.contourArea)
                cv.drawContours(display_image, [largest_inner_contour], -1, (255, 0, 255), 1)
                cv.drawContours(display_image, [largest_contour], -1, (255, 0, 0), 1)

    # Update average FPS every second
    end_cpu = time.time()
    cpu_time = (end_cpu - start_cpu) * 1000
    cpu_times.append(cpu_time)
    print("CPU time : " + str(cpu_time) + "ms")
    current_time = time.time()
    if current_time - last_update_time >= update_interval:
        avg_cpu_time = np.mean(cpu_times)
        avg_cpu_fps = 1000 / avg_cpu_time if avg_cpu_time > 0 else 0

        print("Average CPU FPS : " + str(avg_cpu_fps))
        cpu_times = []
        last_update_time = current_time

    # Display original image with bounding box, average FPS, and average color
    cv.putText(display_image, f"CPU FPS: {avg_cpu_fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.line(display_image, (int(frame_width/2), 0), (int(frame_width/2), int(frame_height)), (0, 255, 0), 2)

    #cv.imshow("Output Image", display_image)
    return display_image

def captureFrame(count):
    ret, captured_image = cap.read()
    if ret:
        cv.imwrite(f"Images/captured/captured ({count}).jpg", captured_image)
        return captured_image
    else:
        print("Failed to capture image")
