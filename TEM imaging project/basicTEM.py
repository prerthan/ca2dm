import os
import cv2
import numpy as np
import skimage
from skimage.feature import peak_local_max

# TEM_folder_path = "/Users/prerthanmunireternam/Desktop/CA2DM/TEM imaging project/TEMimages"
# xyz_folder_path = "/Users/prerthanmunireternam/Desktop/CA2DM/TEM imaging project/xyzFiles"

image_path = "/Users/prerthanmunireternam/Desktop/CA2DM/TEM imaging project/TEM3.jpg"

# # Load and preprocess the TEM image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# # Detect atomic positions
coordinates = peak_local_max(blurred, min_distance=5)

# # Convert to Angstroms (assuming scale: 0.1 nm per pixel)
scale_factor = 1.0  # Adjust based on actual TEM scale
xyz_data = ["{} {:.3f} {:.3f} {:.3f}".format("C", x * scale_factor, y * scale_factor, 0.0) for x, y in coordinates]

# # Empty file first
open("graphene.txt", "w").close()

# # Write to XYZ file
with open("graphene.txt", "w") as f:
     f.write(f"{len(coordinates)}\nGraphene structure from TEM\n")
     f.write("\n".join(xyz_data))
