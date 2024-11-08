import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_highpass_filter(image_path, cutoff=30):
    # Step 1: Load the image and convert to grayscale
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError("Image not found or unable to load")

    # Step 2: Apply Fourier Transform
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)  # Shift the zero frequency to the center

    # Step 3: Create a high-pass filter
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  # Center coordinates
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0  # Create a centered square mask

    # Step 4: Apply the mask to the Fourier-transformed image
    filtered_dft = dft_shift * mask

    # Step 5: Inverse Fourier Transform to return to spatial domain
    filtered_dft_shift = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(filtered_dft_shift)
    img_back = np.abs(img_back)  # Get magnitude (real part)

    # Display the images
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title("Original Image"), plt.axis('off')
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title("High-pass Filtered Image"), plt.axis('off')
    plt.show()

# Use the function
apply_highpass_filter('Images\\captured\\original\\original (20241107_153813).jpg', cutoff=750)

