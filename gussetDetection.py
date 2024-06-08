import cv2 as cv
import numpy as np
import time

# Initialize CUDA
cv.cuda.setDevice(0)

cap = cv.VideoCapture(0)  # Use the webcam

cpu_times = []
last_update_time = time.time()
update_interval = 1  # Update FPS every second
avg_cpu_fps = 0  # Initialize average CPU FPS

while True:
    frame_start_time = time.time()
    
    start_open = time.time()
    success, image = cap.read()
    if not success:
        print("Failed to load video")
        break

    
    end_open = time.time()
    open_time = (end_open - start_open) * 1000
    print("Open time : " + str(open_time) + "ms")

    # CPU operations
    start_cpu = time.time()
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)
    _, cpu_thresholded_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_otsu = cv.GaussianBlur(cpu_thresholded_image, (5, 5), 0)
    canny = cv.Canny(blurred_otsu, 100, 200)
    end_cpu = time.time()

    cpu_time = (end_cpu - start_cpu) * 1000
    cpu_times.append(cpu_time)
    print("CPU time : " + str(cpu_time) + "ms")

    # Find contours and draw the bounding box of the largest contour
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Create a mask for the largest contour
        mask = np.zeros_like(grayscale_image)
        cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)

        # Calculate the average color inside the largest contour
        mean_val = cv.mean(image, mask=mask)
        avg_color = (mean_val[0], mean_val[1], mean_val[2])  # BGR order
        print(f"Average color inside the largest contour: {avg_color}")

        # Draw the average color on the image
        cv.putText(image, f"Avg Color: {avg_color}", (x, y - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

    # Update average FPS every second
    current_time = time.time()
    if current_time - last_update_time >= update_interval:
        avg_cpu_time = np.mean(cpu_times)
        avg_cpu_fps = 1000 / avg_cpu_time if avg_cpu_time > 0 else 0

        print("Average CPU FPS : " + str(avg_cpu_fps))
        cpu_times = []
        last_update_time = current_time

    # Display original image with bounding box, average FPS, and average color
    display_image = cv.putText(image.copy(), f"CPU FPS: {avg_cpu_fps:.2f}", 
                               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow("Output Image", display_image)

    key = cv.waitKey(1)
    if key & 0xFF == ord('x'):
        break

cv.destroyAllWindows()
cap.release()
