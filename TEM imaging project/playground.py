import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import NearestNeighbors

def process_image(image_path, sharpenBool, min_thresh, max_thresh, sharpenPeak):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if sharpenBool:
        sharpen_kernel = np.array([[0, -1, 0], [-1, sharpenPeak, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, sharpen_kernel)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh_img = cv2.threshold(blurred_img, min_thresh, max_thresh, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh_img = cv2.dilate(thresh_img, kernel, iterations=1)
    return thresh_img, img

def detect_bright_spots(thresh_img, minArea, maxArea, minThreshold, maxThreshold):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = minArea
    params.maxArea = maxArea
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.minThreshold = minThreshold
    params.maxThreshold = maxThreshold
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh_img)
    return keypoints

def find_missing_atoms(keypoints, scaling_factor=1.2):
    if len(keypoints) < 2:
        return []

    points = np.array([kp.pt for kp in keypoints])
    print(len(points))

    # Find nearest neighbors (assuming graphene's hexagonal structure)
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    missing_atoms = []

    for i, (x, y) in enumerate(points):
        neighbors = distances[i, 1:]  # Exclude self-distance (first column)
        
        # Compute mean spacing for this atom (adaptive thresholding)
        mean_spacing = np.mean(neighbors)
        adaptive_threshold = mean_spacing * scaling_factor  # Use mean-based threshold

        missing_positions = []

        for j, dist in enumerate(neighbors):
            if dist > adaptive_threshold:  # Compare against adaptive threshold
                # Estimate missing atom position
                neighbor_index = indices[i, j + 1]
                nx, ny = points[neighbor_index]
                mx, my = (x + nx) / 2, (y + ny) / 2  # Midpoint

                missing_positions.append((mx, my))

        missing_atoms.extend(missing_positions)

    print(len(missing_atoms))
    return missing_atoms



def visualize_detection(img, keypoints, missing_atoms):
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_with_keypoints, cmap='gray')
    
    for x, y in missing_atoms:
        plt.scatter(x, y, c='red', s=40, marker='x')
    
    plt.title("Bright Spot Detection - Atoms with Missing Atom Candidates")
    plt.show()

def main():
    image_path = 'TEM4.jpg'
    thresh_img, img = process_image(image_path, sharpenBool=True, min_thresh=100, max_thresh=255, sharpenPeak=5)
    keypoints = detect_bright_spots(thresh_img, minArea=0.01, maxArea=700, minThreshold=0.5, maxThreshold=700)
    missing_atoms = find_missing_atoms(keypoints)
    visualize_detection(img, keypoints, missing_atoms)

if __name__ == "__main__":
    main()
