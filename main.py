#!/usr/bin/env python3

import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QSpinBox, QFrame, QPushButton, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QImage, QPixmap
from PIL import Image
from image_viewer import HighPerformanceImageView
from rectification_window import RectificationWindow
from super_resolution_window import SuperResolutionWindow
from corner_refinement import RefineCornersThread
from scipy.ndimage import gaussian_filter


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
        self.rectify_button.setToolTip(
            "Open secondary window for quadrilateral rectification")
        self.rectify_button.clicked.connect(self.open_rectification_window)

        # Refine Corners button
        self.refine_button = QPushButton("Refine Corners")
        self.refine_button.setToolTip(
            "Automatically refine corner points by minimizing the brightness of the grid lines. Assumes TL, TR, BR, BL order.")
        self.refine_button.clicked.connect(self.start_corner_refinement)

        # Super-resolution button
        self.super_res_button = QPushButton("Super-Resolution")
        self.super_res_button.setToolTip(
            "Open super-resolution window (requires 4 points)")
        self.super_res_button.clicked.connect(
            self.open_super_resolution_window)

        # Add to layout
        controls_layout.addWidget(h_label)
        controls_layout.addWidget(self.h_repetitions)
        controls_layout.addWidget(v_label)
        controls_layout.addWidget(self.v_repetitions)
        controls_layout.addWidget(self.rectify_button)
        controls_layout.addWidget(self.refine_button)
        controls_layout.addWidget(self.super_res_button)
        controls_layout.addStretch()  # Push controls to the left

        layout.addWidget(controls_frame)

        # Add refinement/blur controls
        refine_controls_frame = QFrame()
        refine_controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        refine_layout = QHBoxLayout(refine_controls_frame)

        refine_layout.addWidget(QLabel("Blur Sigma:"))
        self.blur_sigma_spinbox = QDoubleSpinBox()
        self.blur_sigma_spinbox.setMinimum(0.1)
        self.blur_sigma_spinbox.setMaximum(20.0)
        self.blur_sigma_spinbox.setSingleStep(0.5)
        self.blur_sigma_spinbox.setValue(3.0)
        self.blur_sigma_spinbox.setToolTip(
            "Sigma for Gaussian blur before darkness optimization.")
        refine_layout.addWidget(self.blur_sigma_spinbox)

        self.blur_preview_label = QLabel()
        self.blur_preview_label.setFixedSize(64, 64)
        self.blur_preview_label.setStyleSheet("border: 1px solid gray;")
        self.blur_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.blur_preview_label.setText("No Point")
        refine_layout.addWidget(QLabel("Blur Preview (1st Pt):"))
        refine_layout.addWidget(self.blur_preview_label)

        refine_layout.addStretch()
        layout.addWidget(refine_controls_frame)

        # Add a status label for refinement progress
        self.refinement_status_label = QLabel("")
        self.refinement_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.refinement_status_label)

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

        # Set callback for node changes in image viewer to update dependent views
        self.image_viewer.set_node_change_callback(self.on_node_changed)

        # Connect blur controls
        self.blur_sigma_spinbox.valueChanged.connect(self.update_blur_preview)

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
                pil_image = Image.frombytes(
                    "RGBA", (width, height), np.asarray(buffer).tobytes())
                # Convert to RGB for consistency
                return pil_image.convert("RGB")

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

    def on_node_changed(self):
        """Callback for when a node is added or moved."""
        self.update_rectification()
        self.update_blur_preview()

    def update_blur_preview(self):
        """
        Updates the blur preview mini-plot by only blurring a small patch of the
        image around the first corner point for UI responsiveness.
        """
        if not self.image_viewer.nodes:
            self.blur_preview_label.setText("No Point")
            self.blur_preview_label.setPixmap(QPixmap())  # Clear pixmap
            return

        pil_image = self.get_pil_image()
        if pil_image is None:
            return

        image_gray = np.array(pil_image.convert(
            'L')).astype(np.float32) / 255.0
        h, w = image_gray.shape
        sigma = self.blur_sigma_spinbox.value()

        pt0 = self.image_viewer.nodes[0]
        x, y = int(pt0.x()), int(pt0.y())

        # Define the total size of the patch we'll work with, including margins for the blur
        preview_size = 64
        margin = int(np.ceil(4 * sigma))
        patch_size = preview_size + 2 * margin
        half_patch = patch_size // 2

        # Get the patch from the source image, handling edges by padding
        y_start, y_end = y - half_patch, y + half_patch
        x_start, x_end = x - half_patch, x + half_patch

        # Create a patch of zeros and copy the valid image data into it
        source_patch = np.zeros((patch_size, patch_size), dtype=np.float32)

        # Calculate valid source and destination slices
        src_y_start_img = max(0, y_start)
        src_y_end_img = min(h, y_end)
        src_x_start_img = max(0, x_start)
        src_x_end_img = min(w, x_end)

        dst_y_start_patch = src_y_start_img - y_start
        dst_y_end_patch = src_y_end_img - y_start
        dst_x_start_patch = src_x_start_img - x_start
        dst_x_end_patch = src_x_end_img - x_start

        source_patch[dst_y_start_patch:dst_y_end_patch, dst_x_start_patch:dst_x_end_patch] = \
            image_gray[src_y_start_img:src_y_end_img,
                       src_x_start_img:src_x_end_img]

        # Now blur the padded patch
        blurred_patch = gaussian_filter(source_patch, sigma=sigma)

        # The preview is the central 32x32 area of the blurred patch
        final_crop = blurred_patch[margin:margin +
                                   preview_size, margin:margin+preview_size]

        # Convert to QImage/QPixmap for display
        norm_crop = (final_crop * 255).clip(0, 255).astype(np.uint8)
        q_image = QImage(
            norm_crop.data, norm_crop.shape[1], norm_crop.shape[0], norm_crop.shape[1], QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.blur_preview_label.setPixmap(pixmap.scaled(
            self.blur_preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        ))

    def start_corner_refinement(self):
        """Initiates the corner refinement process in a worker thread."""
        pil_image = self.get_pil_image()
        if pil_image is None or len(self.image_viewer.nodes) != 4:
            self.refinement_status_label.setText(
                "Error: Requires an image and 4 corner points.")
            return

        # Disable buttons during refinement
        self.rectify_button.setEnabled(False)
        self.refine_button.setEnabled(False)
        self.super_res_button.setEnabled(False)

        h_reps = self.get_horizontal_repetitions()
        v_reps = self.get_vertical_repetitions()
        sigma = self.blur_sigma_spinbox.value()

        # Run refinement in a separate thread
        self.refinement_thread = RefineCornersThread(
            pil_image, self.image_viewer.nodes, h_reps, v_reps, sigma
        )
        self.refinement_thread.progress.connect(
            self.refinement_status_label.setText)
        self.refinement_thread.finished.connect(self.on_refinement_finished)
        self.refinement_thread.start()

    def on_refinement_finished(self, optimized_corners):
        """Handles the result from the RefineCornersThread."""
        # Convert numpy array back to list of QPointF
        new_nodes = [QPointF(pt[0], pt[1]) for pt in optimized_corners]

        # Update the nodes in the image viewer
        self.image_viewer.nodes = new_nodes
        self.image_viewer.viewport().update()  # Redraw the overlay
        self.update_rectification()  # Update the rectification view

        self.refinement_status_label.setText("Corner refinement complete.")

        # Re-enable buttons
        self.rectify_button.setEnabled(True)
        self.refine_button.setEnabled(True)
        self.super_res_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
