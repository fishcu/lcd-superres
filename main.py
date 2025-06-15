#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QSpinBox, QFrame, QPushButton, QDoubleSpinBox,
                             QGridLayout, QProgressBar, QFileDialog)
from PyQt6.QtCore import Qt, QPointF, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PIL import Image
import cv2
from image_viewer import HighPerformanceImageView
from corner_refinement import RefineCornersThread
from scipy.ndimage import gaussian_filter
from skimage.transform import ProjectiveTransform, warp
from numpy.fft import fft2, fftshift
from pynufft import NUFFT


class SuperResolutionThread(QThread):
    """Worker thread for computing super-resolution to avoid freezing the UI."""
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal(np.ndarray)

    def __init__(self, pil_image, quad_points, h_reps, v_reps, output_size):
        super().__init__()
        self.pil_image = pil_image
        self.quad_points = quad_points
        self.h_reps = h_reps
        self.v_reps = v_reps
        self.output_size = output_size

    def run(self):
        """Execute the super-resolution computation."""
        img_array = np.array(self.pil_image).astype(np.float32) / 255.0
        img_h, img_w, _ = img_array.shape

        # Generate sample coordinates
        self.status_update.emit("Generating sample coordinates...")
        self.progress.emit(10)

        num_patches = self.h_reps * self.v_reps

        u = np.linspace(0.5 / self.output_size, 1 - 0.5 /
                        self.output_size, self.output_size)
        v = np.linspace(0.5 / self.output_size, 1 - 0.5 /
                        self.output_size, self.output_size)
        uv_grid = np.stack(np.meshgrid(u, v), axis=-1).reshape(-1, 2)

        i = np.arange(self.h_reps)
        j = np.arange(self.v_reps)
        ij_grid = np.stack(np.meshgrid(i, j), axis=-1).reshape(-1, 2)

        rectified_grid = uv_grid[:, np.newaxis, :] + ij_grid[np.newaxis, :, :]
        rectified_points = rectified_grid.reshape(-1, 2).astype(np.float32)

        # Transform coordinates to image space
        self.status_update.emit("Transforming coordinates...")
        self.progress.emit(25)

        src_points = np.array([
            [0, 0], [self.h_reps, 0], [self.h_reps, self.v_reps], [0, self.v_reps]
        ], dtype=np.float32)
        dst_points = np.array([[pt.x(), pt.y()]
                              for pt in self.quad_points], dtype=np.float32)
        rectified_to_image_matrix = cv2.getPerspectiveTransform(
            src_points, dst_points)

        orig_points = cv2.perspectiveTransform(
            rectified_points.reshape(-1, 1, 2), rectified_to_image_matrix)
        orig_points = orig_points.reshape(-1, 2)

        # Resample using NUFFT
        self.status_update.emit("Resampling pixels with NUFFT...")
        self.progress.emit(40)

        # Initialize NUFFT
        NufftObj = NUFFT()

        # The NUFFT forward method computes sum(c[k] * exp(-i*om*k)).
        # The inverse FFT is sum(c[k] * exp(i*2*pi*t*k/N)).
        # So, we set om = -2*pi*t/N to match the formula.
        # pynufft expects coordinates in (y, x) order, matching numpy's (row, col) indexing.
        om = np.zeros_like(orig_points)
        om[:, 0] = -2 * np.pi * orig_points[:, 1] / img_h  # y-coordinates
        om[:, 1] = -2 * np.pi * orig_points[:, 0] / img_w  # x-coordinates

        # Plan the NUFFT
        Nd = (img_h, img_w)  # Image dimensions
        Kd = (img_h * 2, img_w * 2)  # Oversampled grid size
        Jd = (6, 6)  # Interpolation neighborhood
        NufftObj.plan(om, Nd, Kd, Jd)

        sampled_pixels_channels = []
        total_channels = img_array.shape[2]
        for ch_idx in range(total_channels):
            self.status_update.emit(
                f"Resampling channel {ch_idx + 1}/{total_channels}...")

            # Take the FFT of the image channel and shift the zero-frequency to the center.
            fourier_coeffs = fftshift(fft2(img_array[:, :, ch_idx]))

            # Evaluate the Fourier series at the non-uniform points.
            y = NufftObj.forward(fourier_coeffs)

            # The result needs to be scaled by 1/N like a standard iFFT.
            resampled_channel = np.real(y) / (img_h * img_w)

            sampled_pixels_channels.append(resampled_channel)

            self.progress.emit(40 + int(60 * (ch_idx + 1) / total_channels))

        sampled_pixels = np.stack(sampled_pixels_channels, axis=1)

        # Compute median across patches
        self.status_update.emit("Computing median...")

        stacked_patches = sampled_pixels.reshape(
            self.output_size, self.output_size, num_patches, 3)
        stacked_patches = np.swapaxes(stacked_patches, 2, 3)
        median_image = np.median(stacked_patches, axis=3)

        self.status_update.emit("Done.")
        self.progress.emit(100)

        self.finished.emit(median_image)


class MainWindow(QMainWindow):
    def __init__(self, image_path="test.tif"):
        super().__init__()
        self.setWindowTitle("Grid repetition super-resolution")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create the main controls area with 3 groups
        controls_container = QWidget()
        controls_container.setMaximumHeight(250)
        controls_layout = QHBoxLayout(controls_container)

        # Left side container for rectification and corner refinement
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        # Group 1: Rectification (top-left)
        rectification_frame = QFrame()
        rectification_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        rectification_layout = QVBoxLayout(rectification_frame)
        rectification_layout.addWidget(QLabel("<b>1. Rectification</b>"))

        # Create horizontal layout for controls on left, previews on right
        rect_content_layout = QHBoxLayout()

        # Left side: Repetition controls (stacked vertically)
        rep_controls_layout = QVBoxLayout()

        # Horizontal repetitions
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("H:"))
        self.h_repetitions = QSpinBox()
        self.h_repetitions.setMinimum(1)
        self.h_repetitions.setMaximum(999)
        self.h_repetitions.setValue(1)
        self.h_repetitions.setToolTip("Horizontal repetitions")
        h_layout.addWidget(self.h_repetitions)
        h_layout.addStretch()
        rep_controls_layout.addLayout(h_layout)

        # Vertical repetitions
        v_layout = QHBoxLayout()
        v_layout.addWidget(QLabel("V:"))
        self.v_repetitions = QSpinBox()
        self.v_repetitions.setMinimum(1)
        self.v_repetitions.setMaximum(999)
        self.v_repetitions.setValue(1)
        self.v_repetitions.setToolTip("Vertical repetitions")
        v_layout.addWidget(self.v_repetitions)
        v_layout.addStretch()
        rep_controls_layout.addLayout(v_layout)

        rep_controls_layout.addStretch()

        # Right side: Rectified previews
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Rectified Previews:"))

        self.rectified_preview_labels = []
        for i in range(4):
            label = QLabel()
            label.setFixedSize(64, 64)
            label.setStyleSheet("border: 1px solid gray;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setText("N/A")
            self.rectified_preview_labels.append(label)
            preview_layout.addWidget(label)

        # Add to horizontal layout
        rect_content_layout.addLayout(rep_controls_layout)
        rect_content_layout.addLayout(preview_layout)

        rectification_layout.addLayout(rect_content_layout)

        # Group 2: Corner Refinement (bottom-left)
        refinement_frame = QFrame()
        refinement_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        refinement_layout = QVBoxLayout(refinement_frame)
        refinement_layout.addWidget(QLabel("<b>2. Corner Refinement</b>"))

        # Create horizontal layout for controls on left, preview on right
        refine_content_layout = QHBoxLayout()

        # Left side: Blur controls (stacked vertically)
        blur_controls_layout = QVBoxLayout()

        # Blur sigma control
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Blur Sigma:"))
        self.blur_sigma_spinbox = QDoubleSpinBox()
        self.blur_sigma_spinbox.setMinimum(0.1)
        self.blur_sigma_spinbox.setMaximum(20.0)
        self.blur_sigma_spinbox.setSingleStep(0.5)
        self.blur_sigma_spinbox.setValue(3.0)
        self.blur_sigma_spinbox.setToolTip(
            "Sigma for Gaussian blur before optimization")
        sigma_layout.addWidget(self.blur_sigma_spinbox)
        sigma_layout.addStretch()
        blur_controls_layout.addLayout(sigma_layout)

        # Refine button
        self.refine_button = QPushButton("Refine Corners")
        self.refine_button.setToolTip("Automatically refine corner points")
        blur_controls_layout.addWidget(self.refine_button)

        blur_controls_layout.addStretch()

        # Right side: Blur preview
        blur_preview_layout = QHBoxLayout()
        blur_preview_layout.addWidget(QLabel("Blur Preview:"))
        self.blur_preview_label = QLabel()
        self.blur_preview_label.setFixedSize(64, 64)
        self.blur_preview_label.setStyleSheet("border: 1px solid gray;")
        self.blur_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.blur_preview_label.setText("No Point")
        blur_preview_layout.addWidget(self.blur_preview_label)
        blur_preview_layout.addStretch()

        # Add to horizontal layout
        refine_content_layout.addLayout(blur_controls_layout)
        refine_content_layout.addLayout(blur_preview_layout)

        refinement_layout.addLayout(refine_content_layout)

        # Status for refinement
        self.refinement_status_label = QLabel("")
        self.refinement_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        refinement_layout.addWidget(self.refinement_status_label)

        # Add left side groups to container
        left_layout.addWidget(rectification_frame)
        left_layout.addWidget(refinement_frame)
        left_layout.addStretch()

        # Group 3: Super-Resolution (right side)
        superres_frame = QFrame()
        superres_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        superres_layout = QVBoxLayout(superres_frame)
        superres_layout.addWidget(QLabel("<b>3. Super-Resolution</b>"))

        # Create horizontal layout for controls on left, image on right
        superres_content_layout = QHBoxLayout()

        # Left side: Controls
        superres_controls_container = QWidget()
        superres_controls_layout = QVBoxLayout(superres_controls_container)

        # Output size control
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Output Size:"))
        self.output_size_spinbox = QSpinBox()
        self.output_size_spinbox.setMinimum(32)
        self.output_size_spinbox.setMaximum(1024)
        self.output_size_spinbox.setValue(128)
        size_layout.addWidget(self.output_size_spinbox)
        size_layout.addStretch()
        superres_controls_layout.addLayout(size_layout)

        # Compute button
        self.compute_button = QPushButton("Compute Super-Resolution")
        self.compute_button.clicked.connect(self.start_superres_computation)
        superres_controls_layout.addWidget(self.compute_button)

        # Save button
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_superres_image)
        self.save_button.setEnabled(False)
        superres_controls_layout.addWidget(self.save_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        superres_controls_layout.addWidget(self.progress_bar)

        # Status label
        self.superres_status_label = QLabel()
        self.superres_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        superres_controls_layout.addWidget(self.superres_status_label)

        superres_controls_layout.addStretch()

        # Right side: Result image preview
        self.superres_image_label = QLabel("Result will appear here")
        self.superres_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.superres_image_label.setStyleSheet(
            "border: 1px solid gray; background-color: #333;")
        self.superres_image_label.setMinimumSize(200, 150)

        # Add controls and image to horizontal layout
        superres_content_layout.addWidget(superres_controls_container)
        superres_content_layout.addWidget(self.superres_image_label)

        superres_layout.addLayout(superres_content_layout)

        # Add to main controls layout with equal stretch factors
        controls_layout.addWidget(left_container, 1)  # 50% width
        controls_layout.addWidget(superres_frame, 1)  # 50% width

        main_layout.addWidget(controls_container)

        # Add image viewer
        self.image_viewer = HighPerformanceImageView()
        main_layout.addWidget(self.image_viewer)

        self.image_viewer.set_image(image_path)

        # Initialize super-resolution variables
        self.median_image = None

        # Connect signals for real-time updates
        self.h_repetitions.valueChanged.connect(self.update_grid)
        self.v_repetitions.valueChanged.connect(self.update_grid)
        self.h_repetitions.valueChanged.connect(self.update_rectified_previews)
        self.v_repetitions.valueChanged.connect(self.update_rectified_previews)

        # Set callback for node changes in image viewer to update dependent views
        self.image_viewer.set_node_change_callback(self.on_node_changed)

        # Connect blur and refinement controls
        self.blur_sigma_spinbox.valueChanged.connect(self.update_blur_preview)
        self.refine_button.clicked.connect(self.start_corner_refinement)

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
        """Updates the four rectified preview patches for corner grid cells."""
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

            # Optimize warp by cropping source image first
            # Find bounding box with padding
            x_min, y_min = np.min(src_coords, axis=0)
            x_max, y_max = np.max(src_coords, axis=0)

            padding = 10
            crop_x_start = int(max(0, x_min - padding))
            crop_y_start = int(max(0, y_min - padding))
            crop_x_end = int(min(img_w, x_max + padding))
            crop_y_end = int(min(img_h, y_max + padding))

            # Crop image and translate coordinates to local patch system
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
        """Updates blur preview by blurring a small patch around the first corner point."""
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
        self.update_blur_preview()  # Update blur preview after refinement

        self.refinement_status_label.setText("Corner refinement complete.")

        # Re-enable buttons
        self.refine_button.setEnabled(True)

    def start_superres_computation(self):
        """Initiates the super-resolution computation in a worker thread."""
        pil_image = self.get_pil_image()
        if pil_image is None or len(self.image_viewer.nodes) != 4:
            self.superres_status_label.setText(
                "Error: Requires an image and 4 corner points.")
            return

        # Disable buttons during computation
        self.compute_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.superres_status_label.setText("Initializing...")
        self.median_image = None

        h_reps = self.get_horizontal_repetitions()
        v_reps = self.get_vertical_repetitions()
        output_size = self.output_size_spinbox.value()

        # Run computation in a separate thread
        self.superres_thread = SuperResolutionThread(
            pil_image, self.image_viewer.nodes, h_reps, v_reps, output_size
        )
        self.superres_thread.progress.connect(
            self.progress_bar.setValue)
        self.superres_thread.status_update.connect(
            self.superres_status_label.setText)
        self.superres_thread.finished.connect(self.on_superres_finished)
        self.superres_thread.start()

    def on_superres_finished(self, median_image):
        """Handles the result from the SuperResolutionThread."""
        self.median_image = median_image

        # Display the result image
        result_image = (median_image * 255).clip(0, 255).astype(np.uint8)
        height, width, channel = result_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(result_image.data, width, height,
                         bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.superres_image_label.setPixmap(pixmap.scaled(
            self.superres_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

        self.progress_bar.setVisible(False)
        self.superres_status_label.setText(
            "Super-resolution computation complete.")
        self.compute_button.setEnabled(True)
        self.save_button.setEnabled(True)

    def save_superres_image(self):
        """Saves the computed super-resolution image to a file."""
        if self.median_image is None:
            self.superres_status_label.setText(
                "Error: No super-resolution image computed.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Super-Resolution Image",
            "",
            "PNG Image (*.png);;16-bit TIFF Image (*.tif *.tiff)"
        )

        if not file_path:
            return

        try:
            # Determine the format from the selected filter or file extension
            is_png = 'png' in selected_filter.lower()
            is_tiff = 'tif' in selected_filter.lower()

            if not is_png and not is_tiff:
                if file_path.lower().endswith('.png'):
                    is_png = True
                elif file_path.lower().endswith(('.tif', '.tiff')):
                    is_tiff = True
                else:  # Default to PNG
                    if '.' not in file_path:
                        file_path += '.png'
                    is_png = True

            if is_png:
                # Save as 8-bit PNG
                uint8_image = (self.median_image * 255).clip(0,
                                                             255).astype(np.uint8)
                pil_img = Image.fromarray(uint8_image, 'RGB')
                pil_img.save(file_path)
            elif is_tiff:
                # Save as 16-bit TIFF using OpenCV
                uint16_image = (self.median_image *
                                65535).clip(0, 65535).astype(np.uint16)
                # cv2.imwrite expects BGR format, so we convert from RGB
                bgr_image = cv2.cvtColor(uint16_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, bgr_image)

            self.superres_status_label.setText(f"Image saved to {file_path}")

        except Exception as e:
            self.superres_status_label.setText(f"Error saving image: {e}")

    def closeEvent(self, event):
        """Ensure threads are stopped on window close."""
        if hasattr(self, 'superres_thread') and self.superres_thread.isRunning():
            self.superres_thread.quit()
            self.superres_thread.wait()
        if hasattr(self, 'refinement_thread') and self.refinement_thread.isRunning():
            self.refinement_thread.quit()
            self.refinement_thread.wait()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Grid repetition super-resolution tool")
    parser.add_argument("image_path", nargs="?", default="test.tif", 
                       help="Path to the input image file (default: test.tif)")
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = MainWindow(args.image_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
