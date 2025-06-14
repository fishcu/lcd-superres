#!/usr/bin/env python3

import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QSpinBox, QFrame, QPushButton)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage
from PIL import Image
from image_viewer import HighPerformanceImageView
from rectification_window import RectificationWindow
from super_resolution_window import SuperResolutionWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("High-Performance Image Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Initialize windows
        self.rectification_window = None
        self.super_resolution_window = None
        
        # Add repetition controls
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_frame.setMaximumHeight(60)
        controls_layout = QHBoxLayout(controls_frame)
        
        # Horizontal repetitions
        h_label = QLabel("Horizontal Repetitions:")
        self.h_repetitions = QSpinBox()
        self.h_repetitions.setMinimum(1)
        self.h_repetitions.setMaximum(999)
        self.h_repetitions.setValue(1)
        self.h_repetitions.setToolTip("Number of horizontal repetitions (≥ 1)")
        
        # Vertical repetitions
        v_label = QLabel("Vertical Repetitions:")
        self.v_repetitions = QSpinBox()
        self.v_repetitions.setMinimum(1)
        self.v_repetitions.setMaximum(999)
        self.v_repetitions.setValue(1)
        self.v_repetitions.setToolTip("Number of vertical repetitions (≥ 1)")
        
        # Rectification button
        self.rectify_button = QPushButton("Open Rectification Window")
        self.rectify_button.setToolTip("Open secondary window for quadrilateral rectification")
        self.rectify_button.clicked.connect(self.open_rectification_window)
        
        # Super-resolution button
        self.super_res_button = QPushButton("Super-Resolution")
        self.super_res_button.setToolTip("Open super-resolution window (requires 4 points)")
        self.super_res_button.clicked.connect(self.open_super_resolution_window)
        
        # Add to layout
        controls_layout.addWidget(h_label)
        controls_layout.addWidget(self.h_repetitions)
        controls_layout.addWidget(v_label)
        controls_layout.addWidget(self.v_repetitions)
        controls_layout.addWidget(self.rectify_button)
        controls_layout.addWidget(self.super_res_button)
        controls_layout.addStretch()  # Push controls to the left
        
        layout.addWidget(controls_frame)
        
        # Add image viewer
        self.image_viewer = HighPerformanceImageView()
        layout.addWidget(self.image_viewer)
        
        self.image_viewer.set_image("test.tif")
        
        # Connect signals for real-time rectification updates
        self.h_repetitions.valueChanged.connect(self.update_rectification)
        self.v_repetitions.valueChanged.connect(self.update_rectification)
        
        # Connect signals for grid updates
        self.h_repetitions.valueChanged.connect(self.update_grid)
        self.v_repetitions.valueChanged.connect(self.update_grid)
        
        # Set callback for node changes in image viewer
        self.image_viewer.set_node_change_callback(self.update_rectification)
        
        # Initialize grid
        self.update_grid()

    def get_horizontal_repetitions(self):
        """Get current horizontal repetitions value."""
        return self.h_repetitions.value()
    
    def get_vertical_repetitions(self):
        """Get current vertical repetitions value."""
        return self.v_repetitions.value()
    
    def get_pil_image(self):
        """Get the original image as a PIL Image."""
        if self.image_viewer.original_image is not None:
            original_image = self.image_viewer.original_image
            width = original_image.width()
            height = original_image.height()
            
            if original_image.format() == QImage.Format.Format_RGB888:
                buffer = original_image.bits().asarray(width * height * 3)
                return Image.frombytes("RGB", (width, height), np.asarray(buffer).tobytes())
            
            elif original_image.format() in [QImage.Format.Format_ARGB32, QImage.Format.Format_RGB32, QImage.Format.Format_RGBA8888]:
                 buffer = original_image.bits().asarray(width * height * 4)
                 pil_image = Image.frombytes("RGBA", (width, height), np.asarray(buffer).tobytes())
                 return pil_image.convert("RGB") # Convert to RGB for consistency
            
        return None
    
    def open_rectification_window(self):
        """Open the rectification window."""
        if self.rectification_window is None:
            self.rectification_window = RectificationWindow(self)
        
        self.rectification_window.show()
        self.rectification_window.raise_()
        self.rectification_window.activateWindow()
        
        # Update the rectification if we have 4 points
        self.update_rectification()
    
    def open_super_resolution_window(self):
        """Open the super-resolution window."""
        if self.super_resolution_window is None:
            self.super_resolution_window = SuperResolutionWindow(self)
        
        self.super_resolution_window.show()
        self.super_resolution_window.raise_()
        self.super_resolution_window.activateWindow()
    
    def update_grid(self):
        """Update the grid overlay in the image viewer."""
        h_reps = self.get_horizontal_repetitions()
        v_reps = self.get_vertical_repetitions()
        self.image_viewer.set_grid_repetitions(h_reps, v_reps)
    
    def update_rectification(self):
        """Update rectification window if open and 4 points available."""
        # Auto-open the rectification window when we have exactly 4 points
        if len(self.image_viewer.nodes) == 4:
            if self.rectification_window is None:
                self.rectification_window = RectificationWindow(self)
            
            if not self.rectification_window.isVisible():
                self.rectification_window.show()
                self.rectification_window.raise_()
                self.rectification_window.activateWindow()
        
        if (self.rectification_window is not None and 
            self.rectification_window.isVisible() and 
            len(self.image_viewer.nodes) == 4):
            
            # Get the original image from the image viewer
            pil_image = self.get_pil_image()
            if pil_image is not None:
                # Get repetition values
                h_reps = self.get_horizontal_repetitions()
                v_reps = self.get_vertical_repetitions()
                
                # Perform rectification
                self.rectification_window.rectify_quadrilateral(
                    pil_image, self.image_viewer.nodes, h_reps, v_reps
                )
        elif (self.rectification_window is not None and 
              self.rectification_window.isVisible() and 
              len(self.image_viewer.nodes) < 4):
            # Clear the rectification window if we don't have 4 points
            self.rectification_window.clear_view()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 