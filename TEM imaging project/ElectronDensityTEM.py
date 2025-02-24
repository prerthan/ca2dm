import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply a sharpening filter
    sharpen_kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, sharpen_kernel)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive thresholding to handle uneven illumination
    thresh_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply Canny edge detection to highlight edges of irregular shapes
    edges = cv2.Canny(blurred_img, 50, 150)

    return thresh_img, edges, img

def detect_blobs(thresh_img, edges):
    # Combine thresholded image with edges to enhance irregular shapes
    combined = cv2.bitwise_or(thresh_img, edges)

    # Find contours in the combined image
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    keypoints = []
    for cnt in contours:
        # Calculate area to filter small noise
        area = cv2.contourArea(cnt)
        if area > 5:  # Adjust this threshold based on your dataset
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                keypoints.append(cv2.KeyPoint(cx, cy, np.sqrt(area)))
    return keypoints, contours

def generate_xyz(keypoints, output_file="xyz.txt"):
    with open(output_file, 'w') as file:
        file.write(f"{len(keypoints)}\n")
        file.write("Graphene lattice\n")
        for kp in keypoints:
            x, y = kp.pt  # Get coordinates
            print(f"{x},{y}")
            file.write(f"C {x} {y} 0.0\n")  # Assuming z=0
    print(f"XYZ file generated: {output_file}")

def visualize_detection(img, keypoints, contours):
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title("Blob Detection - Atoms")

    plt.subplot(1, 2, 2)
    plt.imshow(img_with_contours, cmap='gray')
    plt.title("Contour Detection - Irregular Shapes")

    plt.show()

def main():
    # Main execution
    image_path = 'TEM4.jpg'  # Replace with your TEM image
    thresh_img, edges, img = process_image(image_path)
    keypoints, contours = detect_blobs(thresh_img, edges)

    # Generate XYZ file
    generate_xyz(keypoints)

    # Visualize results
    visualize_detection(img, keypoints, contours)

if __name__ == "__main__":
    main()


