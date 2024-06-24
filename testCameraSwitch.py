import cv2
import time

# Function to initialize webcam with given resolution
def initialize_webcam(width, height, backend=cv2.CAP_DSHOW):
    cap = cv2.VideoCapture(0, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

# Set resolutions
display_width, display_height = 640, 480
capture_width, capture_height = 3840, 2160

# Open the webcam with low resolution using DirectShow backend
cap = initialize_webcam(display_width, display_height)

# Initialize variables
last_capture_time = time.time()

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame in low resolution
        cv2.imshow("Webcam (Low Resolution)", frame)
        print(frame.shape)

        # Capture high-resolution image every 5 seconds
        current_time = time.time()
        if current_time - last_capture_time >= 5:
            # Switch to high resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
            time.sleep(0.01)  # Allow the camera to adjust

            # Capture the high-resolution frame
            ret, frame_high = cap.read()
            if ret:
                timestamp = int(time.time())
                filename = f"high_res_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame_high)
                print(f"Captured high resolution image: {filename}")
                print(frame_high.shape)


            # Switch back to low resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
            time.sleep(0.01)  # Allow the camera to adjust

            last_capture_time = current_time

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
