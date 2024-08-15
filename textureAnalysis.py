import cv2 as cv
import numpy as np
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

def entropy_analysis(image):
    # Reduce the disk radius for faster processing
    disk_radius = 5  # Smaller disk for faster computation
    entr_img = entropy(image, disk(disk_radius))

    # Normalize the entropy image to the range [0, 1]
    entr_img = (entr_img - entr_img.min()) / (entr_img.max() - entr_img.min())

    # Convert the normalized entropy image to 8-bit
    img_8bit = img_as_ubyte(entr_img)

    return img_8bit

