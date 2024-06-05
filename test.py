import asyncio
import cv2
import numpy as np
from winrt.windows.media.capture import MediaCapture
from winrt.windows.media.media_properties import ImageEncodingProperties
from winrt.windows.storage.streams import InMemoryRandomAccessStream
from PIL import Image

async def capture_photo():
    media_capture = MediaCapture()
    await media_capture.initialize_async()
    
    # Create an in-memory stream to hold the photo
    stream = InMemoryRandomAccessStream()
    
    # Set the resolution
    image_properties = ImageEncodingProperties.create_jpeg()
    image_properties.width = 3024
    image_properties.height = 4032
    
    # Capture the photo to the stream
    await media_capture.capture_photo_to_stream_async(image_properties, stream)
    
    # Get the bytes from the stream
    stream.seek(0)
    data = await stream.read_async(stream.size)
    data = bytearray(data)
    
    # Convert bytes to image
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save the image
    cv2.imwrite('high_res_image.jpg', image)

# Run the async function
asyncio.run(capture_photo())