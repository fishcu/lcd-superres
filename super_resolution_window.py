import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QPushButton, QSpinBox, QProgressBar, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PIL import Image
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

        # 1. Generate all sample coordinates in the original image space
        self.status_update.emit("Generating sample coordinates...")
        self.progress.emit(10)
        
        num_patches = self.h_reps * self.v_reps
        
        u = np.linspace(0.5 / self.output_size, 1 - 0.5 / self.output_size, self.output_size)
        v = np.linspace(0.5 / self.output_size, 1 - 0.5 / self.output_size, self.output_size)
        uv_grid = np.stack(np.meshgrid(u, v), axis=-1).reshape(-1, 2)

        i = np.arange(self.h_reps)
        j = np.arange(self.v_reps)
        ij_grid = np.stack(np.meshgrid(i, j), axis=-1).reshape(-1, 2)
        
        rectified_grid = uv_grid[:, np.newaxis, :] + ij_grid[np.newaxis, :, :]
        rectified_points = rectified_grid.reshape(-1, 2).astype(np.float32)

        # 2. Transform coordinates from rectified to image space
        self.status_update.emit("Transforming coordinates...")
        self.progress.emit(25)

        src_points = np.array([
            [0, 0], [self.h_reps, 0], [self.h_reps, self.v_reps], [0, self.v_reps]
        ], dtype=np.float32)
        dst_points = np.array([[pt.x(), pt.y()] for pt in self.quad_points], dtype=np.float32)
        rectified_to_image_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        orig_points = cv2.perspectiveTransform(rectified_points.reshape(-1, 1, 2), rectified_to_image_matrix)
        orig_points = orig_points.reshape(-1, 2)

        # 3. Resample using NUFFT
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
            self.status_update.emit(f"Resampling channel {ch_idx + 1}/{total_channels}...")
            
            # Take the FFT of the image channel and shift the zero-frequency to the center.
            fourier_coeffs = fftshift(fft2(img_array[:, :, ch_idx]))
            
            # Evaluate the Fourier series at the non-uniform points.
            y = NufftObj.forward(fourier_coeffs)
            
            # The result needs to be scaled by 1/N like a standard iFFT.
            resampled_channel = np.real(y) / (img_h * img_w)

            sampled_pixels_channels.append(resampled_channel)

            self.progress.emit(40 + int(60 * (ch_idx + 1) / total_channels))

        sampled_pixels = np.stack(sampled_pixels_channels, axis=1)

        # 4. Compute Median
        self.status_update.emit("Computing median...")
        
        # Reshape to (output_size, output_size, num_patches, 3) for median calculation
        stacked_patches = sampled_pixels.reshape(self.output_size, self.output_size, num_patches, 3)
        
        # Swap axes to (output_size, output_size, 3, num_patches) to compute median over patches
        stacked_patches = np.swapaxes(stacked_patches, 2, 3)
        median_image = np.median(stacked_patches, axis=3)
        
        self.status_update.emit("Done.")
        self.progress.emit(100)
        
        self.finished.emit(median_image)


class SuperResolutionWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.median_image = None
        self.setWindowTitle("Super-Resolution")
        self.setGeometry(300, 300, 800, 650)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Output Size:"))
        self.output_size_spinbox = QSpinBox()
        self.output_size_spinbox.setMinimum(32)
        self.output_size_spinbox.setMaximum(1024)
        self.output_size_spinbox.setValue(128)
        controls_layout.addWidget(self.output_size_spinbox)
        
        self.compute_button = QPushButton("Compute Super-Resolution")
        self.compute_button.clicked.connect(self.start_computation)
        controls_layout.addWidget(self.compute_button)

        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        controls_layout.addWidget(self.save_button)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Image display
        self.image_label = QLabel("Click 'Compute' to generate super-resolution image.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #333;")
        layout.addWidget(self.image_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    def start_computation(self):
        # Disable button during computation
        self.compute_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing...")
        self.median_image = None
        
        # Get data from main window
        image_viewer = self.main_window.image_viewer
        pil_image = self.main_window.get_pil_image()
        quad_points = image_viewer.nodes
        h_reps = self.main_window.get_horizontal_repetitions()
        v_reps = self.main_window.get_vertical_repetitions()
        output_size = self.output_size_spinbox.value()
        
        if pil_image is None or len(quad_points) != 4:
            self.status_label.setText("Error: Requires an image and 4 points in the main window.")
            self.compute_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return
            
        # Run computation in a separate thread
        self.thread = SuperResolutionThread(pil_image, quad_points, h_reps, v_reps, output_size)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.status_update.connect(self.status_label.setText)
        self.thread.finished.connect(self.on_computation_finished)
        self.thread.start()

    def on_computation_finished(self, median_image):
        """Display the result when computation is done."""
        self.median_image = median_image
        result_image = (self.median_image * 255).clip(0, 255).astype(np.uint8)

        height, width, channel = result_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))
        
        self.progress_bar.setVisible(False)
        self.compute_button.setEnabled(True)
        self.save_button.setEnabled(True)

    def save_image(self):
        if self.median_image is None:
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
                else: # Default to PNG
                    if '.' not in file_path:
                        file_path += '.png'
                    is_png = True

            if is_png:
                # Save as 8-bit PNG
                uint8_image = (self.median_image * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(uint8_image, 'RGB')
                pil_img.save(file_path)
            elif is_tiff:
                # Save as 16-bit TIFF using OpenCV
                uint16_image = (self.median_image * 65535).clip(0, 65535).astype(np.uint16)
                # cv2.imwrite expects BGR format, so we convert from RGB
                bgr_image = cv2.cvtColor(uint16_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, bgr_image)

            self.status_label.setText(f"Image saved to {file_path}")

        except Exception as e:
            self.status_label.setText(f"Error saving image: {e}")

    def closeEvent(self, event):
        """Ensure thread is stopped on window close."""
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        super().closeEvent(event) 