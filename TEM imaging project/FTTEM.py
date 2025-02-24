import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the TEM image
image_path = "TEM3.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Preprocess the image (enhance contrast, reduce noise)
image = cv2.equalizeHist(image)  # Histogram equalization for contrast enhancement
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Slight blur to reduce noise

# Step 2: Perform Fourier Transform on the image
f = np.fft.fft2(image_blurred)
fshift = np.fft.fftshift(f)  # Shift zero frequency component to the center

# Step 3: Visualize the magnitude spectrum of the Fourier transform (optional)
magnitude_spectrum = np.abs(fshift)
magnitude_spectrum_log = np.log(1 + magnitude_spectrum)

plt.figure(figsize=(10, 10))
plt.imshow(magnitude_spectrum_log, cmap='gray')
plt.title('Magnitude Spectrum of the Fourier Transform')
plt.axis('off')
plt.show()

# Step 4: Create a Bandpass Filter for detecting periodic structures
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Create a circular mask for the bandpass filter (preserve low and high frequencies)
mask = np.zeros((rows, cols), np.uint8)

# Set the low and high frequency radii based on periodicity
radius_low = 20  # Lower cutoff radius (low frequency filter)
radius_high = 100  # Upper cutoff radius (high frequency filter)

# Create circular mask to preserve the periodic structure frequencies
cv2.circle(mask, (ccol, crow), radius_high, 1, thickness=-1)
cv2.circle(mask, (ccol, crow), radius_low, 0, thickness=-1)

# Step 5: Apply the Bandpass Filter in the frequency domain
fshift_filtered = fshift * mask

# Step 6: Perform Inverse Fourier Transform to get the filtered image in spatial domain
f_ishift = np.fft.ifftshift(fshift_filtered)
image_filtered = np.fft.ifft2(f_ishift)
image_filtered = np.abs(image_filtered)

# Step 7: Normalize the filtered image for better visualization
image_filtered = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Step 8: Detect peaks or blobs in the filtered image (atomic positions)
# Use simple thresholding for peak detection
_, thresh = cv2.threshold(image_filtered, 50, 255, cv2.THRESH_BINARY)

# Find contours (blobs corresponding to atomic positions)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 9: Visualize the detected atomic positions
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw detected contours (atomic positions)
for contour in contours:
    if cv2.contourArea(contour) > 10:  # Filter small areas
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(output_image, (cX, cY), 5, (0, 255, 0), -1)  # Green circle for atomic positions

# Step 10: Show the final image with detected atomic positions
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.title('Detected Atomic Positions on TEM Image')
plt.axis('off')
plt.show()
