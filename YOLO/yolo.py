
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import time
 
cap = cv.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video
 
 
#image_path = "YOLO/images/in/gusset (1).jpg"
model = YOLO('../weights/best.pt')
 
classNames = ["gusset"]
 
prev_frame_time = 0
new_frame_time = 0
 
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
 
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
 
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
 
    cv.imshow("Image", img)
     # Wait  'x' key to exit
    key = cv.waitKey(5)
    if key == ord('x'):
        break
# Release resources
cv.destroyAllWindows()
