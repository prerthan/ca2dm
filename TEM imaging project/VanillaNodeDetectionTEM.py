import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path, sharpenBool, min, max, sharpenPeak):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", img)

    # Apply a sharpening filter to enhance details
    if(sharpenBool): 
        sharpen_kernel = np.array([[0, -1, 0], [-1, sharpenPeak, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, sharpen_kernel)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply binary thresholding (NOT inverted) to keep bright spots 200,255
    #_, thresh_img = cv2.threshold(blurred_img, 150, 255, cv2.THRESH_BINARY)
    _, thresh_img = cv2.threshold(blurred_img, min, max, cv2.THRESH_BINARY)
    
    # Use morphological operations to enhance bright spots
    kernel = np.ones((3, 3), np.uint8)
    thresh_img = cv2.dilate(thresh_img, kernel, iterations=1)  # Expands bright spots

    return thresh_img, img

def detect_bright_spots(thresh_img, minArea, maxArea, minThreshold, maxThreshold):
    # Set up the SimpleBlobDetector parameters to detect bright spots
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255  # Detect bright (white) spots

    params.filterByArea = True
    #params.minArea = 0.1  # Adjust based on atom size
    #params.maxArea = 500  # Avoid large noise artifacts 100
    params.minArea = minArea
    params.maxArea = maxArea

    params.filterByCircularity = False  # Atoms might not be perfectly circular
    params.filterByConvexity = False  # Avoid strict shape filtering

    # params.minThreshold = 1
    # params.maxThreshold = 500
    params.minThreshold = minThreshold
    params.maxThreshold = maxThreshold

    # Create a SimpleBlobDetector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect bright spots in the thresholded image
    keypoints = detector.detect(thresh_img)

    return keypoints

def generate_xyz(keypoints, output_file="graphene.xyz"):
    with open(output_file, 'w') as file:
        file.write(f"{len(keypoints)}\n")
        file.write("Graphene lattice\n")
        for kp in keypoints:
            x, y = kp.pt  # Get coordinates
            file.write(f"C {x} {y} 0.0\n")  # Assuming z=0
    print(f"XYZ file generated: {output_file}")

def visualize_detection(img, keypoints):
    # Draw detected bright spots as green dots
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title("Bright Spot Detection - Atoms")
    plt.show()


def main():
    # Main execution
    image_path = 'TEM4.jpg'  # Replace with your TEM image
    thresh_img, img = process_image(image_path, sharpenBool=True, min=100, max=255, sharpenPeak=5)
    keypoints = detect_bright_spots(thresh_img, minArea=0.01, maxArea=700, minThreshold=0.5, maxThreshold=550)

    # Visualize detected atoms
    visualize_detection(img, keypoints)

    # Generate XYZ file
    generate_xyz(keypoints)

if __name__ == "__main__":
    main()


