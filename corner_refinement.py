import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from scipy.optimize import minimize, Bounds
from scipy.ndimage import map_coordinates, gaussian_filter
import cv2

class RefineCornersThread(QThread):
    """
    Worker thread for refining the four corner points of a quadrilateral.

    This thread optimizes the corner positions by minimizing the total brightness
    of the pixels that lie along the grid lines defined by the corners.
    The image is first blurred to create a smoother optimization landscape.
    The optimization is constrained to a small search area around the initial points.
    """
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, pil_image, initial_corners, h_reps, v_reps, sigma, parent=None):
        super().__init__(parent)
        image_gray = np.array(pil_image.convert('L')).astype(np.float32) / 255.0
        
        # Pre-blur the image to create smoother valleys for the optimizer
        self.image_blurred = gaussian_filter(image_gray, sigma=sigma)
        
        self.initial_corners = np.array([[p.x(), p.y()] for p in initial_corners], dtype=np.float32)
        self.h_reps = h_reps
        self.v_reps = v_reps
        self.samples_per_line = 30 # Number of samples to take along each grid line

    def calculate_darkness_score(self, flat_corners):
        """
        Calculates a score based on the total brightness of pixels
        lying underneath the grid lines. A lower score (darker lines) is better.
        """
        corners = flat_corners.reshape(4, 2)
        p0, p1, p2, p3 = corners[0], corners[1], corners[2], corners[3]

        all_lines = []
        # Add outer boundary lines
        all_lines.extend([(p0, p1), (p1, p2), (p2, p3), (p3, p0)])

        # Add horizontal subdivision lines
        for i in range(1, self.v_reps):
            t = i / self.v_reps
            start_point = p0 * (1 - t) + p3 * t
            end_point = p1 * (1 - t) + p2 * t
            all_lines.append((start_point, end_point))

        # Add vertical subdivision lines
        for i in range(1, self.h_reps):
            t = i / self.h_reps
            start_point = p0 * (1 - t) + p1 * t
            end_point = p3 * (1 - t) + p2 * t
            all_lines.append((start_point, end_point))

        # Generate sample points along all lines
        all_sample_points = []
        for start, end in all_lines:
            x_points = np.linspace(start[0], end[0], self.samples_per_line)
            y_points = np.linspace(start[1], end[1], self.samples_per_line)
            all_sample_points.append(np.vstack([y_points, x_points])) # map_coordinates expects (row, col)

        if not all_sample_points:
            return 0

        coords = np.hstack(all_sample_points)
        
        # Sample the brightness values from the pre-blurred image
        brightness_values = map_coordinates(self.image_blurred, coords, order=1, mode='nearest')
        
        score = np.sum(brightness_values)
        
        return score

    def run(self):
        """Execute the corner refinement optimization."""
        self.progress.emit("Starting refinement...")
        
        initial_guess = self.initial_corners.flatten()
        
        pixel_bounds = 5
        bounds = Bounds(initial_guess - pixel_bounds, initial_guess + pixel_bounds)

        result = minimize(
            self.calculate_darkness_score,
            initial_guess,
            method='Powell',
            bounds=bounds,
            options={'disp': False, 'ftol': 1e-4, 'xtol': 1e-5}
        )
        
        if result.success:
            optimized_corners = result.x.reshape(4, 2)
            self.progress.emit("Refinement successful.")
            self.finished.emit(optimized_corners)
        else:
            self.progress.emit(f"Refinement failed: {result.message}")
            self.finished.emit(self.initial_corners) 