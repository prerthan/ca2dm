import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
# from matplotlib.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QMessageBox)

class CSVPlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Plotter with Radius")
        # self.setGeometry(100, 100, 800, 600)
        self.setGeometry(100, 50, 100, 2000)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Add button to load CSV
        self.load_button = QPushButton("Load CSV File")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        # Label for status
        self.label = QLabel("Load a CSV file to visualize the points with radius.")
        layout.addWidget(self.label)

        # Set up Matplotlib figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def load_csv(self):
        # Open file dialog to load CSV
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            try:
                data = pd.read_csv(file_name)
                self.plot_data(data)
            except Exception as e:
                self.show_error(f"Error loading file:\n{e}")

    def plot_data(self, data):
        self.ax.clear()  # Clear previous plots

        # Check for expected columns
        if 'x' in data.columns and 'y' in data.columns and 'radius' in data.columns:
            x = data['x'].values
            y = data['y'].values
            radius = data['radius'].values

            # Plot each point as a circle using the radius
            for i in range(len(x)):
                circle = plt.Circle((x[i], y[i]), radius[i], color='red', fill=True, alpha=0.5)
                self.ax.add_patch(circle)  # Add circle to the plot

                # Use a fixed distance threshold for connecting edges
                for j in range(i + 1, len(x)):
                    distance = np.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2)
                    if distance < 15:  # Use a fixed threshold value (adjust as needed)
                        self.ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', linewidth=0.5)  # Lines between neighbors

            self.ax.set_title("2D Plot of Points with Radius")
            self.ax.set_xlabel("X Coordinate")
            self.ax.set_ylabel("Y Coordinate")
            self.ax.axis('equal')  # Keep aspect ratio constant
            self.canvas.draw()  # Refresh the canvas

        else:
            self.show_error("The CSV file must contain 'x', 'y', and 'radius' columns.")

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CSVPlotterApp()
    window.show()
    sys.exit(app.exec_())