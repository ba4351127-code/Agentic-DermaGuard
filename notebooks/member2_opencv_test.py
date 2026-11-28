
import cv2
import numpy as np

# Step 1: Create a sample image
image = np.zeros((400, 400, 3), dtype=np.uint8)

# Step 2: Add a sample skin-like colored region
image[:] = (180, 190, 210)

# Step 3: Print image information
print("Image created successfully!")

height, width, channels = image.shape

print("Height:", height)
print("Width:", width)
print("Channels:", channels)

# Step 4: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Grayscale conversion successful!")

# Step 5: Resize image
resized = cv2.resize(image, (224, 224))

print("Resized image:", resized.shape)

# Step 6: Calculate brightness
brightness = np.mean(gray)

print("Average brightness:", brightness)

# Step 7: Calculate blur score
blur_score = cv2.Laplacian(
    gray, cv2.CV_64F
).var()

print("Blur score:", blur_score)

print("OpenCV basic test completed!")
