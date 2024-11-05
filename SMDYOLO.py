import cv2 as cv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Function to set resolution
def set_resolution(width, height):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    print(f"Resolution set to: {width}x{height}")

# Function to start video capture
def start_capture():
    global cap
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # Use DirectShow backend

    # Check if camera opened successfully
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video stream.")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # Adding a delay for camera to initialize
    import time
    time.sleep(1)

    # Read a few frames to stabilize the feed
    for _ in range(5):
        ret, frame = cap.read()

    # Update the video feed in the GUI
    update_frame()

def update_frame():
    global cap, video_label

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture video.")
        return

    # Get the current resolution
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Add resolution text to the frame
    cv.putText(frame, f'Resolution: {width}x{height}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the frame to RGB format for displaying in Tkinter
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Resize the image while maintaining the aspect ratio
    max_width = 640  # Set max width for the display
    max_height = 480  # Set max height for the display
    aspect_ratio = width / height

    if width > height:
        new_width = min(max_width, width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(max_height, height)
        new_width = int(new_height * aspect_ratio)

    frame_resized = cv.resize(frame_rgb, (new_width, new_height))

    # Create a PhotoImage object from the resized image
    img = Image.fromarray(frame_resized)
    img_tk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = img_tk  # Keep a reference to avoid garbage collection
    video_label.configure(image=img_tk)

    video_label.after(10, update_frame)  # Schedule the next frame update

# Main function to set up the GUI
def main():
    global cap, video_label
    cap = None

    root = tk.Tk()
    root.title("Video Capture")

    # Create a label to display the video
    video_label = tk.Label(root)
    video_label.pack()

    start_button = tk.Button(root, text="Start Capture", command=start_capture)
    start_button.pack(pady=20)

    # Create buttons for different resolutions
    res_640x480 = tk.Button(root, text="Set Resolution 640x480", command=lambda: set_resolution(640, 480))
    res_1280x720 = tk.Button(root, text="Set Resolution 1280x720", command=lambda: set_resolution(1280, 720))
    res_3840x2160 = tk.Button(root, text="Set Resolution 3840x2160", command=lambda: set_resolution(3840, 2160))

    res_640x480.pack(pady=5)
    res_1280x720.pack(pady=5)
    res_3840x2160.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
