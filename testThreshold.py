import cv2
print(cv2.__version__)

# Test loading and displaying an image
img = cv2.imread('test_image.jpg')  # Replace with a valid image path
cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
