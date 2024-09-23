import cv2

# Callback function for the trackbars
def adjust_brightness_contrast(val):
    pass

# Open webcam
cap = cv2.VideoCapture(0)

# Create a window for trackbars
cv2.namedWindow('Webcam Preview')

# Create trackbars for brightness and contrast adjustments
cv2.createTrackbar('Brightness', 'Webcam Preview', 50, 100, adjust_brightness_contrast)
cv2.createTrackbar('Contrast', 'Webcam Preview', 50, 100, adjust_brightness_contrast)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get current positions of brightness and contrast trackbars
    brightness = cv2.getTrackbarPos('Brightness', 'Webcam Preview')
    contrast = cv2.getTrackbarPos('Contrast', 'Webcam Preview')
    
    # Adjust brightness and contrast
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast/50, beta=brightness-50)
    
    # Display the resulting frame
    cv2.imshow('Webcam Preview', adjusted_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
