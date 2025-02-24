import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QLineEdit
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import ast
from sklearn.cluster import KMeans
import numpy as np
import math

class GraphPlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize file paths
        self.node_file = ''
        self.edge_file = ''
        self.nodes_df = None
        self.edges_df = None

        # Set up the main window
        self.setWindowTitle("Graph Plotter")
        self.setGeometry(100, 100, 800, 600)

        # Set up the layout and widgets
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)

        self.status_label = QLabel("Please load both CSV files.", self)
        self.layout.addWidget(self.status_label)

        self.node_btn = QPushButton("Load Nodes CSV", self)
        self.node_btn.clicked.connect(self.load_nodes_csv)
        self.layout.addWidget(self.node_btn)

        self.edge_btn = QPushButton("Load Edges CSV", self)
        self.edge_btn.clicked.connect(self.load_edges_csv)
        self.layout.addWidget(self.edge_btn)

        self.plot_btn = QPushButton("Plot Graph", self)
        self.plot_btn.clicked.connect(self.plot_graph)
        self.layout.addWidget(self.plot_btn)

        self.cluster_btn = QPushButton("Detect Clusters", self)
        self.cluster_btn.clicked.connect(self.detect_clusters)
        self.cluster_btn.setEnabled(False)  # Initially disabled until graph is plotted
        self.layout.addWidget(self.cluster_btn)

        self.histogram_btn = QPushButton("Show Histograms", self)
        self.histogram_btn.clicked.connect(self.show_histograms)
        self.histogram_btn.setEnabled(False)  # Initially disabled until graph is plotted
        self.layout.addWidget(self.histogram_btn)

        self.size_of_clusters = QLineEdit(self)
        self.size_of_clusters.setPlaceholderText('how big each cluster is (e.g., 6)')
        self.layout.addWidget(self.size_of_clusters)

        # Setup canvas for matplotlib
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.central_widget)

    def load_nodes_csv(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select Nodes CSV file", "", "CSV Files (*.csv)", options=options)
        if file:
            self.node_file = file
            self.update_status(f"Nodes CSV loaded: {self.node_file}")

    def load_edges_csv(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select Edges CSV file", "", "CSV Files (*.csv)", options=options)
        if file:
            self.edge_file = file
            self.update_status(f"Edges CSV loaded: {self.edge_file}")
            self.histogram_btn.setEnabled(True)

    def update_status(self, message):
        self.status_label.setText(message)

    def plot_graph(self):
        if not self.node_file or not self.edge_file:
            self.update_status("Error: Please load both CSV files.")
            return

        try:
            # Load CSV files
            nodes_df = pd.read_csv(self.node_file)
            edges_df = pd.read_csv(self.edge_file)

            # Clear the previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Plot nodes
            for _, node in nodes_df.iterrows():
                x, y, radius = node['x'], node['y'], node['radius']
                ax.plot(x, y, 'bo', markersize=(radius * 2), alpha=0.5)
                #ax.text(x, y, f"({x}, {y})", fontsize=8, ha='right')

            # Plot edges
            for _, edge in edges_df.iterrows():
                node1 = ast.literal_eval(edge['Node1'])
                node2 = ast.literal_eval(edge['Node2'])
                ax.plot([node1[0], node2[0]], [node1[1], node2[1]], 'r-')

            ax.set_aspect('equal', 'box')
            ax.set_title("Graph Visualization")
            ax.set_xlabel("X-coordinate")
            ax.set_ylabel("Y-coordinate")

            # Refresh the canvas
            self.canvas.draw()

            self.cluster_btn.setEnabled(True)
            self.update_status("Graph plotted successfully.")

        except Exception as e:
            self.update_status(f"Error plotting graph: {e}")

    def show_histograms(self):
        if not self.edge_file:
            self.update_status("Error: Please load Edges CSV file.")
            return

        try:
            # Load edges data
            edges_df = pd.read_csv(self.edge_file)
            lengths = []
            angles = []

            # Calculate lengths and angles
            edge_coordinates = []
            for _, edge in edges_df.iterrows():
                node1 = ast.literal_eval(edge['Node1'])
                node2 = ast.literal_eval(edge['Node2'])
                edge_coordinates.append((node1, node2))
                
                # Calculate length of the edge
                length = math.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2)
                lengths.append(length)
                
                # Calculate angle if it's not the first edge
                if len(edge_coordinates) > 1:
                    # Get the previous edge
                    prev_node1, prev_node2 = edge_coordinates[-2]

                    # Calculate angle between this edge and the previous edge
                    vec1 = np.array([node2[0] - node1[0], node2[1] - node1[1]])
                    vec2 = np.array([prev_node2[0] - prev_node1[0], prev_node2[1] - prev_node1[1]])
                    
                    angle = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
                    angles.append(np.degrees(angle))  # Convert to degrees

            # Clear the figure for new plots
            self.figure.clear()

            # Plot lengths histogram
            ax1 = self.figure.add_subplot(211)  # 2 rows, 1 column, 1st subplot
            ax1.hist(lengths, bins=20, color='blue', alpha=0.7)
            ax1.set_title('Histogram of Edge Lengths')
            ax1.set_xlabel('Length')
            ax1.set_ylabel('Frequency')

            # Plot angles histogram
            ax2 = self.figure.add_subplot(212)  # 2 rows, 1 column, 2nd subplot
            ax2.hist(angles, bins=20, color='green', alpha=0.7)
            ax2.set_title('Histogram of Angles Between Edges')
            ax2.set_xlabel('Angle (degrees)')
            ax2.set_ylabel('Frequency')

            # Refresh the canvas
            self.canvas.draw()
            self.update_status("Histograms plotted successfully.")

        except Exception as e:
            self.update_status(f"Error generating histograms: {e}")

    def detect_clusters(self):
            if not self.node_file or not self.edge_file:
                self.update_status("Error: Please load both CSV files.")
                return

            try:
                nodes_df = pd.read_csv(self.node_file)
                N = len(nodes_df)
                number_of_centroid = math.floor( N / int(self.size_of_clusters.text()))
                input = max(number_of_centroid,1)
                coordinates = nodes_df[['x', 'y']].values

                # Update: Using KMeans as an example
                kmeans = KMeans(n_clusters=input)
                labels = kmeans.fit_predict(coordinates)

                ax = self.figure.gca()
                unique_labels = set(labels)
                colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

                for k, col in zip(unique_labels, colors):
                    class_member_mask = (labels == k)
                    xy = coordinates[class_member_mask]
                    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgewidth=1, markersize=12, alpha=0.7)

                ax.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 'x', markerfacecolor='k', markersize=14)

                self.canvas.draw()
                self.update_status("Clusters detected and highlighted on the graph.")

            except Exception as e:
                self.update_status(f"Error detecting clusters: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphPlotterApp()
    window.show()
    sys.exit(app.exec_())