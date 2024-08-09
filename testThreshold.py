import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

# Replace 'your_image_path.jpg' with the path to your image
image_path = 'a.jpg'

# Load your image
image = imread(image_path, as_gray=True)

# If the image is not in 8-bit, convert it
if image.dtype != np.uint8:
    image = img_as_ubyte(image)

# Calculate entropy on your image
entr_img = entropy(image, disk(10))

# Plotting the results
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)

ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title("Image")
ax0.axis("off")
fig.colorbar(ax0.imshow(image, cmap=plt.cm.gray), ax=ax0)

ax1.imshow(entr_img, cmap='viridis')
ax1.set_title("Local Entropy")
ax1.axis("off")
fig.colorbar(ax1.imshow(entr_img, cmap='viridis'), ax=ax1)

fig.tight_layout()
plt.show()
