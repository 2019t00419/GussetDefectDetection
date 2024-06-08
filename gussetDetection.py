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
captured=False

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
        #x, y, w, h = cv.boundingRect(largest_contour)
        #cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rect = cv.minAreaRect(largest_contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image,[box],0,(0,0,255),2)
        (x,y),(MA,ma),angle = cv.fitEllipse(largest_contour)
        cv.putText(image, ("Major axis length : "+str(int(MA))+"    Minor axis length : "+str(int(ma))), ((10, 40)),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        
        # Create a mask for the largest contour
        mask = np.zeros_like(grayscale_image)
        cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)
        cv.drawContours(canny, [largest_contour], -1, 0, 2)

        # Mask the Canny image
        masked_canny = cv.bitwise_and(canny, canny, mask=mask)

        # Find contours inside the masked canny image
        inner_contours, _ = cv.findContours(masked_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        largest_inner_contour = None
        if inner_contours:
            largest_inner_contour = max(inner_contours, key=cv.contourArea)
            #x_inner, y_inner, w_inner, h_inner = cv.boundingRect(largest_inner_contour)
            #cv.rectangle(image, (x_inner, y_inner), (x_inner + w_inner, y_inner + h_inner), (255, 0, 0), 2)

            # Create a mask for the largest inner contour
            mask_inner = np.zeros_like(grayscale_image)
            cv.drawContours(mask_inner, [largest_inner_contour], -1, 255, thickness=cv.FILLED)

            # Calculate the average color inside the largest inner contour
            mean_val_inner = cv.mean(image, mask=mask_inner)
            avg_color_inner = (mean_val_inner[0], mean_val_inner[1], mean_val_inner[2])  # BGR order
            print(f"Average color inside the largest inner contour: {avg_color_inner}")

            # Draw the average color on the image
            #cv.putText(image, f"Avg Color: {avg_color_inner}", (x_inner, y_inner - 10),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

    # Update average FPS every second
    current_time = time.time()
    if current_time - last_update_time >= update_interval:
        avg_cpu_time = np.mean(cpu_times)
        avg_cpu_fps = 1000 / avg_cpu_time if avg_cpu_time > 0 else 0

        print("Average CPU FPS : " + str(avg_cpu_fps))
        cpu_times = []
        last_update_time = current_time

    # Display original image with bounding box, average FPS, and average color
    display_image = cv.putText(image.copy(), f"CPU FPS: {avg_cpu_fps:.2f}",(10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.drawContours(display_image, [largest_contour], -1, (255,0,0), 1)
    
    frame_height, frame_width, channels = image.shape
    cv.line(display_image, (int(frame_width/2),0), (int(frame_width/2), int(frame_height)), (0,255,0), 2)
    if largest_inner_contour is not None:
        cv.drawContours(display_image, [largest_inner_contour], -1, (255,0,255), 1)
        
        M = cv.moments(largest_contour)


        cx = int((M['m10'] / M['m00']))
        cy = int((M['m01'] / M['m00']))

        cv.circle(display_image, (cx, cy), 5, (255, 0, 0), cv.FILLED)

        cv.imshow("Output Image", display_image)  
        if(cx>(frame_width/2) and not captured):  
            cv.imwrite("Output Image.jpg", display_image)


    key = cv.waitKey(1)
    if key & 0xFF == ord('x'):
        break

cv.destroyAllWindows()
cap.release()
