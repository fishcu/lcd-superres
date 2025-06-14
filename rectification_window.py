import cv2
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PIL import Image

class RectificationWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rectified View")
        self.setGeometry(200, 200, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        self.image_label.setMinimumSize(400, 300)
        layout.addWidget(self.image_label)
        
        # Status label
        self.status_label = QLabel("Select 4 points in the main window to see rectified view")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
    def rectify_quadrilateral(self, image, quad_points, h_reps, v_reps):
        """Rectify quadrilateral region into rectangle."""
        try:
            # Convert PIL image to numpy array if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
                
            # Convert QPointF to numpy array
            src_points = np.array([[pt.x(), pt.y()] for pt in quad_points], dtype=np.float32)
            
            # Calculate the desired output dimensions based on repetitions
            # We'll use a base size and scale it by the repetition ratios
            base_size = 400  # Base dimension
            output_width = int(base_size * h_reps / max(h_reps, v_reps))
            output_height = int(base_size * v_reps / max(h_reps, v_reps))
            
            # Ensure minimum size
            output_width = max(output_width, 200)
            output_height = max(output_height, 200)
            
            # Define destination points for the rectangle
            dst_points = np.array([
                [0, 0],
                [output_width - 1, 0],
                [output_width - 1, output_height - 1],
                [0, output_height - 1]
            ], dtype=np.float32)
            
            # Calculate perspective transformation matrix
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective transformation
            rectified = cv2.warpPerspective(img_array, transform_matrix, (output_width, output_height))
            
            # Convert back to QImage for display
            if len(rectified.shape) == 3:
                height, width, channel = rectified.shape
                bytes_per_line = 3 * width
                q_image = QImage(rectified.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                height, width = rectified.shape
                bytes_per_line = width
                q_image = QImage(rectified.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            
            # Display the rectified image
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale pixmap to fit the label while maintaining aspect ratio
            label_size = self.image_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, 
                                        Qt.TransformationMode.SmoothTransformation)
            
            self.image_label.setPixmap(scaled_pixmap)
            self.status_label.setText(f"Rectified view ({output_width}x{output_height}) - Aspect ratio {h_reps}:{v_reps}")
            
        except Exception as e:
            self.status_label.setText(f"Error during rectification: {str(e)}")
            self.image_label.clear()
    
    def clear_view(self):
        """Clear the rectified view."""
        self.image_label.clear()
        self.status_label.setText("Select 4 points in the main window to see rectified view") 