#!/usr/bin/env python3

import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QSpinBox, QFrame, QPushButton, QDoubleSpinBox,
                             QGridLayout)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QImage, QPixmap
from PIL import Image
from image_viewer import HighPerformanceImageView
from super_resolution_window import SuperResolutionWindow
from corner_refinement import RefineCornersThread
from scipy.ndimage import gaussian_filter
from skimage.transform import ProjectiveTransform, warp


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("High-Performance Image Viewer")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Initialize windows
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
        controls_layout.addWidget(self.refine_button)
        controls_layout.addWidget(self.super_res_button)
        controls_layout.addStretch()  # Push controls to the left

        layout.addWidget(controls_frame)

        # --- Controls for refinement, blur, and rectified previews ---
        secondary_controls_frame = QFrame()
        secondary_controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        secondary_controls_layout = QHBoxLayout(secondary_controls_frame)

        # Blur controls
        secondary_controls_layout.addWidget(QLabel("Blur Sigma:"))
        self.blur_sigma_spinbox = QDoubleSpinBox()
        self.blur_sigma_spinbox.setMinimum(0.1)
        self.blur_sigma_spinbox.setMaximum(20.0)
        self.blur_sigma_spinbox.setSingleStep(0.5)
        self.blur_sigma_spinbox.setValue(3.0)
        self.blur_sigma_spinbox.setToolTip(
            "Sigma for Gaussian blur before darkness optimization.")
        secondary_controls_layout.addWidget(self.blur_sigma_spinbox)

        secondary_controls_layout.addWidget(QLabel("Blur Preview (1st Pt):"))
        self.blur_preview_label = QLabel()
        self.blur_preview_label.setFixedSize(64, 64)
        self.blur_preview_label.setStyleSheet("border: 1px solid gray;")
        self.blur_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.blur_preview_label.setText("No Point")
        secondary_controls_layout.addWidget(self.blur_preview_label)

        # Rectified preview display
        rectified_preview_frame = QFrame()
        rectified_preview_layout = QGridLayout(rectified_preview_frame)
        rectified_preview_layout.addWidget(
            QLabel("Rectified Previews:"), 0, 0, 1, 4)

        self.rectified_preview_labels = []
        for i in range(4):
            label = QLabel()
            label.setFixedSize(64, 64)
            label.setStyleSheet("border: 1px solid gray;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setText("N/A")
            self.rectified_preview_labels.append(label)
            rectified_preview_layout.addWidget(label, 1, i)
        rectified_preview_frame.setLayout(
            rectified_preview_layout)  # Set layout on the frame
        secondary_controls_layout.addWidget(rectified_preview_frame)

        secondary_controls_layout.addStretch()
        layout.addWidget(secondary_controls_frame)

        # Add a status label for refinement progress
        self.refinement_status_label = QLabel("")
        self.refinement_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.refinement_status_label)

        # Add image viewer
        self.image_viewer = HighPerformanceImageView()
        layout.addWidget(self.image_viewer)

        self.image_viewer.set_image("test.tif")

        # Connect signals for real-time updates
        self.h_repetitions.valueChanged.connect(self.update_grid)
        self.v_repetitions.valueChanged.connect(self.update_grid)
        self.h_repetitions.valueChanged.connect(self.update_rectified_previews)
        self.v_repetitions.valueChanged.connect(self.update_rectified_previews)

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

    def get_numpy_image(self):
        """Get the original image as a NumPy array."""
        if self.image_viewer.original_image is None:
            return None

        original_image = self.image_viewer.original_image
        if original_image.isNull():
            return None

        # Convert QImage to a format that can be easily used with NumPy
        if original_image.format() != QImage.Format.Format_RGBA8888:
            original_image = original_image.convertToFormat(
                QImage.Format.Format_RGBA8888)

        width = original_image.width()
        height = original_image.height()

        ptr = original_image.constBits()
        ptr.setsize(height * width * 4)  # The buffer size must be correct
        # Create a NumPy array from the buffer, then copy
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        return arr.copy()

    def get_pil_image(self):
        """Get the original image as a PIL Image."""
        numpy_image = self.get_numpy_image()
        if numpy_image is not None:
            # Convert RGBA to RGB for PIL
            return Image.fromarray(numpy_image).convert("RGB")
        return None

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

    def on_node_changed(self):
        """Callback for when a node is added or moved."""
        self.update_rectified_previews()
        self.update_blur_preview()

    def update_rectified_previews(self):
        """
        Updates the four rectified preview patches for the corner-most grid cells.
        """
        if len(self.image_viewer.nodes) != 4:
            for label in self.rectified_preview_labels:
                label.setText("N/A")
                label.setPixmap(QPixmap())
            return

        image_np = self.get_numpy_image()
        if image_np is None:
            return

        h_reps = self.get_horizontal_repetitions()
        v_reps = self.get_vertical_repetitions()

        tl, tr, br, bl = [np.array([p.x(), p.y()])
                          for p in self.image_viewer.nodes]

        # Define destination rectangle
        dst_coords = np.array([
            [0, 0],
            [64, 0],
            [64, 64],
            [0, 64]
        ])

        # Define which corner of the grid corresponds to which preview label
        preview_indices = {
            0: (0, 0),                  # Top-left preview -> top-left grid cell
            # Top-right preview -> top-right grid cell
            1: (h_reps - 1, 0),
            # Bottom-left preview -> bottom-left grid cell
            2: (0, v_reps - 1),
            # Bottom-right preview -> bottom-right grid cell
            3: (h_reps - 1, v_reps - 1)
        }

        img_h, img_w, _ = image_np.shape

        for i, (u_idx, v_idx) in preview_indices.items():
            # Calculate the four corners of the source quadrilateral using bilinear interpolation
            u0, v0 = u_idx / h_reps, v_idx / v_reps
            u1, v1 = (u_idx + 1) / h_reps, (v_idx + 1) / v_reps

            p00 = tl * (1 - u0) * (1 - v0) + tr * u0 * (1 - v0) + \
                bl * (1 - u0) * v0 + br * u0 * v0
            p10 = tl * (1 - u1) * (1 - v0) + tr * u1 * (1 - v0) + \
                bl * (1 - u1) * v0 + br * u1 * v0
            p11 = tl * (1 - u1) * (1 - v1) + tr * u1 * (1 - v1) + \
                bl * (1 - u1) * v1 + br * u1 * v1
            p01 = tl * (1 - u0) * (1 - v1) + tr * u0 * (1 - v1) + \
                bl * (1 - u0) * v1 + br * u0 * v1

            src_coords = np.array([
                [p00[0], p00[1]],
                [p10[0], p10[1]],
                [p11[0], p11[1]],
                [p01[0], p01[1]]
            ])

            # --- Optimized warp by cropping the source image ---

            # Find the bounding box of the source quadrilateral, add padding, and clip to image bounds
            x_min, y_min = np.min(src_coords, axis=0)
            x_max, y_max = np.max(src_coords, axis=0)

            padding = 10
            crop_x_start = int(max(0, x_min - padding))
            crop_y_start = int(max(0, y_min - padding))
            crop_x_end = int(min(img_w, x_max + padding))
            crop_y_end = int(min(img_h, y_max + padding))

            # Crop the image and translate the source coordinates to the patch's local system
            source_patch_np = image_np[crop_y_start:crop_y_end,
                                       crop_x_start:crop_x_end]
            translated_src_coords = src_coords - \
                np.array([crop_x_start, crop_y_start])

            # Perform rectification on the smaller patch
            tform = ProjectiveTransform()
            tform.estimate(dst_coords, translated_src_coords)
            patch = warp(source_patch_np, tform,
                         output_shape=(64, 64), mode='edge')

            patch_uint8 = (patch * 255).astype(np.uint8)

            # Display the rectified patch
            h, w, ch = patch_uint8.shape
            q_img = QImage(patch_uint8.data, w, h, ch * w,
                           QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(q_img)
            self.rectified_preview_labels[i].setPixmap(pixmap)

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

        # Define a 64x64 bounding box for the patch to blur
        patch_size = 64
        x_start = max(0, x - patch_size // 2)
        y_start = max(0, y - patch_size // 2)
        x_end = min(w, x_start + patch_size)
        y_end = min(h, y_start + patch_size)

        patch = image_gray[y_start:y_end, x_start:x_end]
        blurred_patch = gaussian_filter(patch, sigma=sigma)

        # Convert back to QImage for display
        blurred_patch_uint8 = (blurred_patch * 255).astype(np.uint8)
        h_patch, w_patch = blurred_patch_uint8.shape
        q_img = QImage(blurred_patch_uint8.data, w_patch,
                       h_patch, w_patch, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.blur_preview_label.setPixmap(
            pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio))

    def start_corner_refinement(self):
        """Initiates the corner refinement process in a worker thread."""
        pil_image = self.get_pil_image()
        if pil_image is None or len(self.image_viewer.nodes) != 4:
            self.refinement_status_label.setText(
                "Error: Requires an image and 4 corner points.")
            return

        # Disable buttons during refinement
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
        self.update_rectified_previews()  # Update the rectified previews

        self.refinement_status_label.setText("Corner refinement complete.")

        # Re-enable buttons
        self.refine_button.setEnabled(True)
        self.super_res_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
