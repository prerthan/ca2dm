import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QSlider, QSpinBox, QDoubleSpinBox, QRadioButton,
                           QButtonGroup)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import io, color, filters, feature
from skimage.util import img_as_ubyte

class AtomDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atom Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.image = None
        self.gray_image = None
        self.blobs = None
        self.circles = []
        self.removed_circles = set()
        self.mode = 'remove'  # Default mode
        
        self.init_ui()
        
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        #layout = QHBoxLayout(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)
        
        # Add control elements
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        control_layout.addWidget(load_button)
        
        # Mode selection
        mode_group = QWidget()
        mode_layout = QHBoxLayout(mode_group)
        self.remove_radio = QRadioButton("Remove Mode")
        self.add_radio = QRadioButton("Add Mode")
        self.remove_radio.setChecked(True)
        mode_layout.addWidget(self.remove_radio)
        mode_layout.addWidget(self.add_radio)
        
        # Connect radio buttons
        self.remove_radio.toggled.connect(self.mode_changed)
        self.add_radio.toggled.connect(self.mode_changed)
        
        control_layout.addWidget(mode_group)
        
        # Parameter controls
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        
        # max edge length
        max_edge_length_label = QLabel("Max Edge Length: ")
        self.max_edge_length = QDoubleSpinBox()
        self.max_edge_length.setRange(0.0, 50.0)
        self.max_edge_length.setValue(20)
        self.max_edge_length.setSingleStep(0.1)
        param_layout.addWidget(max_edge_length_label)
        param_layout.addWidget(self.max_edge_length)

        # max edge length
        min_edge_length_label = QLabel("Min Edge Length: ")
        self.min_edge_length = QDoubleSpinBox()
        self.min_edge_length.setRange(0.0, 50.0)
        self.min_edge_length.setValue(0)
        self.min_edge_length.setSingleStep(0.1)
        param_layout.addWidget(min_edge_length_label)
        param_layout.addWidget(self.min_edge_length)

        # Circle size for manual addition
        circle_size_label = QLabel("Manual Circle Size:")
        self.circle_size_spin = QDoubleSpinBox()
        self.circle_size_spin.setRange(0.1, 20.0)
        self.circle_size_spin.setValue(3.0)
        self.circle_size_spin.setSingleStep(0.1)
        param_layout.addWidget(circle_size_label)
        param_layout.addWidget(self.circle_size_spin)
        
        # Gaussian blur sigma
        blur_label = QLabel("Gaussian Blur (σ):")
        self.blur_spin = QDoubleSpinBox()
        self.blur_spin.setRange(0.1, 10.0)
        self.blur_spin.setValue(2.0)
        self.blur_spin.setSingleStep(0.1)
        param_layout.addWidget(blur_label)
        param_layout.addWidget(self.blur_spin)
        
        # Blob detection parameters
        min_sigma_label = QLabel("Min Sigma:")
        self.min_sigma_spin = QDoubleSpinBox()
        self.min_sigma_spin.setRange(0.1, 10.0)
        self.min_sigma_spin.setValue(2.0)
        param_layout.addWidget(min_sigma_label)
        param_layout.addWidget(self.min_sigma_spin)
        
        max_sigma_label = QLabel("Max Sigma:")
        self.max_sigma_spin = QDoubleSpinBox()
        self.max_sigma_spin.setRange(0.1, 20.0)
        self.max_sigma_spin.setValue(6.0)
        param_layout.addWidget(max_sigma_label)
        param_layout.addWidget(self.max_sigma_spin)
        
        threshold_label = QLabel("Detection Threshold:")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.001, 0.1)
        self.threshold_spin.setValue(0.013)
        self.threshold_spin.setSingleStep(0.001)
        param_layout.addWidget(threshold_label)
        param_layout.addWidget(self.threshold_spin)
        
        # Add detect button
        detect_button = QPushButton("Detect Atoms")
        detect_button.clicked.connect(self.detect_atoms)
        param_layout.addWidget(detect_button)
        
        # Add export button
        export_button = QPushButton("Export Coordinates")
        export_button.clicked.connect(self.export_coordinates)
        param_layout.addWidget(export_button)
        
        control_layout.addWidget(param_widget)
        
        # Create matplotlib canvas
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.canvas.mpl_connect 
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Create additional matplotlib canvas for neighboring nodes plot
        self.figure_neighbors = Figure(figsize=(8, 4))
        self.canvas_neighbors = FigureCanvas(self.figure_neighbors)
        self.ax_neighbors = self.figure_neighbors.add_subplot(111)
        
        # Add widgets to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.canvas_neighbors)

    def mode_changed(self):
        self.mode = 'remove' if self.remove_radio.isChecked() else 'add'
        
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", 
                                                 "Images (*.png *.jpg *.jpeg *.tif)")
        if file_name:
            self.image = io.imread(file_name)
            self.gray_image = color.rgb2gray(self.image) if self.image.ndim == 3 else self.image
            if self.blobs is None:
                self.blobs = np.empty((0, 3))  # Initialize empty blobs array
            self.update_plot()
            
    def detect_atoms(self):
        if self.gray_image is None:
            return
            
        # Get parameters from UI
        sigma = self.blur_spin.value()
        min_sigma = self.min_sigma_spin.value()
        max_sigma = self.max_sigma_spin.value()
        threshold = self.threshold_spin.value()
        
        # Process image
        blurred_image = filters.gaussian(self.gray_image, sigma=sigma)
        self.blobs = feature.blob_log(blurred_image, 
                                    min_sigma=min_sigma,
                                    max_sigma=max_sigma,
                                    num_sigma=10,
                                    threshold=threshold)
        self.removed_circles = set()  # Reset removed circles
        self.count_neighbors()  # Call this after detecting atoms
        self.calculate_intensities()
        #new
        self.category_counts, self.intensity_bins = self.categorize_intensities()
        self.update_plot()
        self.update_neighbors_plot()

    def update_neighbors_plot(self):
        if self.neighbor_counts is not None:
            self.ax_neighbors.clear()  # Clear previous plot
            # Plot the number of neighbors against blob index
            blob_indices = range(len(self.neighbor_counts))
            self.ax_neighbors.bar(blob_indices, self.neighbor_counts, color='blue')
            self.ax_neighbors.set_xlabel('Blob Index')
            self.ax_neighbors.set_ylabel('Number of Neighbors')
            self.ax_neighbors.set_title('Number of Neighbors for Each Blob')
            self.canvas_neighbors.draw()

    def categorize_intensities(self):
        if self.normalized_intensities is None:
            return
        
        # Define new bins for intensity categories (5 categories)
        intensity_bins = np.linspace(0, 1, num=6)  # 5 categories
        categories = np.digitize(self.normalized_intensities, intensity_bins)   # Get category indices
        
        # Count neighbors per category
        category_counts = np.zeros(len(intensity_bins) - 1)
        
        for i in range(len(self.neighbor_counts)):
            if i < len(categories):  # Ensure we're within the bounds of categories
                category_index = categories[i] - 1  # Adjust index for 0-based
                if category_index < len(category_counts):
                    category_counts[category_index] += self.neighbor_counts[i]


        return category_counts, intensity_bins



    def calculate_intensities(self):
        if self.blobs is None:
            return
        
        # Collecting intensities for normalization
        intensities = []
        
        for blob in self.blobs:
            y, x, r = blob
            # Define a circular mask for the blob
            mask = np.zeros(self.gray_image.shape, dtype=bool)
            rr, cc = np.ogrid[:self.gray_image.shape[0], :self.gray_image.shape[1]]
            mask[(rr - y) ** 2 + (cc - x) ** 2 <= (r * np.sqrt(2)) ** 2] = True
            
            # Calculate average intensity within the mask
            average_intensity = np.mean(self.gray_image[mask])
            intensities.append(average_intensity)
        
        # Store normalized intensities
        self.normalized_intensities = None
        if intensities:
            avg_intensity = np.mean(intensities)
            self.normalized_intensities = [i / avg_intensity for i in intensities] 
            #print(self.normalized_intensities)

            # Call intensity categorization
            self.category_counts, self.intensity_bins = self.categorize_intensities()

    def update_neighbors_plot(self):
        if hasattr(self, 'category_counts') and self.category_counts is not None:
            self.ax_neighbors.clear()  # Clear previous plot
            # Plot number of neighbors by intensity category
            categories = [f"{self.intensity_bins[i]:.2f} - {self.intensity_bins[i + 1]:.2f}" for i in range(len(self.intensity_bins) - 1)]
            self.ax_neighbors.bar(categories, self.category_counts, color='blue')
            self.ax_neighbors.set_xlabel('Intensity Categories')
            self.ax_neighbors.set_ylabel('Number of Neighbors')
            self.ax_neighbors.set_title('Number of Neighbors by Intensity Category')
            self.ax_neighbors.set_xticklabels(categories, rotation=45)  # Rotate for better readability

            # Adding debug output for confirmation
            print(f"Category Counts: {self.category_counts}")

            self.canvas_neighbors.draw()

        
    def count_neighbors(self, threshold=20):
        if self.blobs is None:
            return
            
        # Create a list to hold the number of neighbors for each blob
        neighbor_counts = []
        
        for i, blob in enumerate(self.blobs):
            y1, x1, _ = blob
            count = 0
            for j, other_blob in enumerate(self.blobs):
                if i != j:  # Avoid counting the blob itself
                    y2, x2, _ = other_blob
                    distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)  # Euclidean distance
                    if distance <= threshold:  # Check if within threshold
                        count += 1
            neighbor_counts.append(count)
        self.neighbor_counts = neighbor_counts  # Store the counts in the instance variable

    def update_plot(self):
        self.ax.clear()
        if self.gray_image is not None:
            self.ax.imshow(self.gray_image, cmap='gray')
            
            if self.blobs is not None:
                self.circles = []
                for i, blob in enumerate(self.blobs):
                    if i not in self.removed_circles:
                        y, x, r = blob
                        circle = plt.Circle((x, y), r * np.sqrt(2),
                                         color='red', fill=False, linewidth=1.5)
                        self.ax.add_patch(circle)
                        self.circles.append((i, circle))

                         # Add text annotation for the normalized intensity
                        # if self.normalized_intensities is not None:
                        #     intensity_text = f"{self.normalized_intensities[i]:.2f}"  # Format to 2 decimal places
                        #     self.ax.text(x, y, intensity_text, color='white', fontsize=8, 
                        #               verticalalignment='center', horizontalalignment='center')
                            
                        # if self.neighbor_counts is not None:
                        #     neighbor_text = f"N: {self.neighbor_counts[i]}"  # Neighbor count
                        #     self.ax.text(x, y + 10, neighbor_text, color='yellow', fontsize=3,
                        #                 verticalalignment='center', horizontalalignment='center')
            
        self.ax.set_axis_off()
        self.canvas.draw()
        
    def on_click(self, event):
        if event.inaxes != self.ax or self.gray_image is None:
            return
            
        click_x, click_y = event.xdata, event.ydata
        
        if self.mode == 'remove':
            # Remove mode: remove existing circles
            if self.blobs is not None:
                for i, blob in enumerate(self.blobs):
                    if i not in self.removed_circles:
                        y, x, r = blob
                        distance = np.sqrt((x - click_x)**2 + (y - click_y)**2)
                        if distance <= r * np.sqrt(2):
                            self.removed_circles.add(i)
                            self.update_plot()
                            break
        else:
            # Add mode: add new circle
            r = self.circle_size_spin.value()
            new_blob = np.array([[click_y, click_x, r]])
            if self.blobs is None:
                self.blobs = new_blob
            else:
                self.blobs = np.vstack([self.blobs, new_blob])
            self.update_plot()
                    
    def export_coordinates(self):
        upper_limit = self.max_edge_length.value()
        lower_limit = self.min_edge_length.value()
        if self.blobs is None or len(self.blobs) == 0:
            return
            
        # Determine remaining blobs
        remaining_blobs = [blob for i, blob in enumerate(self.blobs) 
                        if i not in self.removed_circles]
                        
        if remaining_blobs:
            # Create DataFrame for nodes
            nodes_df = pd.DataFrame(remaining_blobs, columns=['y', 'x', 'radius'])
            
            # Prepare a list to store edges as coordinate tuples
            edges = []
            
            # Find and add edges based on proximity threshold
            for i, (y1, x1, _) in enumerate(remaining_blobs):
                for j, (y2, x2, r2) in enumerate(remaining_blobs):
                    if i < j:  # Avoid repeating edges
                        distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

                        mask = np.zeros(self.gray_image.shape, dtype=bool)
                        rr, cc = np.ogrid[:self.gray_image.shape[0], :self.gray_image.shape[1]]
                        r = r2
                        mask[(rr - y2) ** 2 + (cc - x2) ** 2 <= (r * np.sqrt(2)) ** 2] = True
                        localised_intensity = np.mean(self.gray_image[mask])
                        #print(localised_intensity)
                        # find point and double check if intensity is above a threshold
                        if (lower_limit <= distance <= upper_limit) and (localised_intensity > 0):
                            #print(f"accepted : {localised_intensity}")
                            # Store edges as tuple of coordinates
                            edges.append(((x1, y1), (x2, y2)))

            # Create DataFrame for edges with each edge as a tuple of coordinates
            edges_df = pd.DataFrame(edges, columns=['Node1', 'Node2'])
            
            # Save nodes and edges to CSVs
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Coordinates", "", "CSV Files (*.csv)")
            
            if file_name:
                # Save nodes
                nodes_df.to_csv(f"{file_name}_nodes.csv", index=False)
                
                # Save edges
                edges_df.to_csv(f"{file_name}_edges.csv", index=False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AtomDetectorApp()
    window.show()
    sys.exit(app.exec_())