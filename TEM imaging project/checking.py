import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read the XYZ file and extract atomic positions
def read_xyz(file_path):
    coordinates = []
    with open(file_path, "r") as f:
        lines = f.readlines()[2:]  # Skip the header lines
        for line in lines:
            parts = line.split()
            x, y = float(parts[1]), float(parts[2])
            coordinates.append((x, y))
    
    return coordinates

# Function to overlay atomic positions on the original TEM image
def overlay_atoms_on_image(image_path, xyz_file):
    # Load the original image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Read atomic positions from the XYZ file
    coordinates = read_xyz(xyz_file)
    
    # Convert the image to RGB for plotting (matplotlib requires RGB for color)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Plotting using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb, cmap='gray', origin='upper')  # Display the original image
    
    # Extract x and y coordinates of the atoms
    x_coords, y_coords = zip(*coordinates)
    
    # Overlay atomic positions (dots) on the image
    plt.scatter(x_coords, y_coords, color='red', s=7)  # Red dots for atomic positions
    
    # Display the combined image
    plt.title('Overlay of Atomic Positions on TEM Image')
    plt.axis('off')  # Turn off axis
    plt.show()

# Example usage
overlay_atoms_on_image("TEM3.jpg", "graphene.txt")
