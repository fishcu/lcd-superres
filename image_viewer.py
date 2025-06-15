from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor
import numpy as np
from PIL import Image


class HighPerformanceImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Image and zoom state
        self.current_zoom = 1.0
        self.image_item = None
        self.original_image = None

        # Node overlay data
        self.nodes = []  # QPointF in scene coordinates
        self.dragging_idx = None
        self.drag_offset = QPointF(0, 0)
        self.middle_button_panning = False
        self.last_pan_point = QPointF(0, 0)

        # Callback for node changes
        self.node_change_callback = None

        # Grid overlay settings
        self.grid_h_reps = 1
        self.grid_v_reps = 1
        self.show_grid = True

    def set_image(self, image_path):
        """Load and display image from file path."""
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        np_image = np.array(pil_image)

        height, width, channel = np_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(np_image.data, width, height,
                         bytes_per_line, QImage.Format.Format_RGB888)
        self.original_image = q_image
        self.display_image(q_image)

    def display_image(self, q_image):
        """Display QImage in the view."""
        if self.image_item:
            self.scene.removeItem(self.image_item)

        pixmap = QPixmap.fromImage(q_image)
        self.image_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        self.current_zoom = 1.0
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        """Handle mouse wheel zooming."""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        old_pos = self.mapToScene(event.position().toPoint())

        factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.current_zoom *= factor
        self.scale(factor, factor)

        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        # Disable smoothing for pixel-perfect viewing when zoomed in
        smooth = self.current_zoom <= 1.0
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)

    def resizeEvent(self, event):
        """Handle window resize."""
        if self.image_item and self.current_zoom <= 1.0:
            self.fitInView(self.scene.sceneRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.MouseButton.MiddleButton:
            self.middle_button_panning = True
            self.last_pan_point = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        node_idx = self.node_at(scene_pos)
        if event.button() == Qt.MouseButton.LeftButton and node_idx is not None:
            self.dragging_idx = node_idx
            self.drag_offset = self.nodes[self.dragging_idx] - scene_pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton and len(self.nodes) < 4:
            self.add_node(scene_pos)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.middle_button_panning:
            delta = event.position() - self.last_pan_point
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self.last_pan_point = event.position()
            return

        if self.dragging_idx is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.nodes[self.dragging_idx] = scene_pos + self.drag_offset
            self.viewport().update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self.middle_button_panning:
            self.middle_button_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if self.dragging_idx is not None:
            self.dragging_idx = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.viewport().update()
            # Trigger callback after node movement
            if self.node_change_callback:
                self.node_change_callback()
            return
        super().mouseReleaseEvent(event)

    def drawForeground(self, painter, rect):
        """Draw zoom-independent overlays."""
        painter.save()
        painter.resetTransform()

        # Draw grid first (behind nodes and selection)
        if self.show_grid and self.image_item:
            self.draw_grid(painter)

        base_line_width = 3.0
        base_node_radius = 8.0

        if len(self.nodes) > 0:
            view_nodes = [self.mapFromScene(node) for node in self.nodes]

            # White background lines
            white_pen = QPen(QColor(255, 255, 255), base_line_width * 2.0)
            painter.setPen(white_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            for i in range(len(view_nodes) - 1):
                painter.drawLine(view_nodes[i], view_nodes[i+1])
            if len(view_nodes) == 4:
                painter.drawLine(view_nodes[3], view_nodes[0])

            # Colored lines on top
            color = QColor(255, 140, 0) if len(
                self.nodes) < 4 else QColor(0, 120, 255)
            colored_pen = QPen(color, base_line_width)
            painter.setPen(colored_pen)

            for i in range(len(view_nodes) - 1):
                painter.drawLine(view_nodes[i], view_nodes[i+1])
            if len(view_nodes) == 4:
                painter.drawLine(view_nodes[3], view_nodes[0])

        for idx, node in enumerate(self.nodes):
            view_pos = self.mapFromScene(node)
            self.draw_node(painter, view_pos, base_node_radius,
                           selected=(idx == self.dragging_idx))

        painter.restore()

    def draw_node(self, painter, pos, radius, selected=False):
        """Draw node with white background and selection indicator."""
        painter.save()

        # White background circle
        white_pen = QPen(Qt.GlobalColor.white, 2.0)
        painter.setPen(white_pen)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QRectF(pos.x() - (radius + 2), pos.y() - (radius + 2),
                                   2 * (radius + 2), 2 * (radius + 2)))

        # Main node circle
        black_pen = QPen(Qt.GlobalColor.black, 1.0)
        painter.setPen(black_pen)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QRectF(pos.x() - radius, pos.y() - radius,
                                   2 * radius, 2 * radius))

        if selected:
            selection_pen = QPen(QColor(255, 200, 0), 2.0)
            painter.setPen(selection_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(pos.x() - (radius + 4), pos.y() - (radius + 4),
                                       2 * (radius + 4), 2 * (radius + 4)))

        painter.restore()

    def node_at(self, scene_pos):
        """Return index of node under scene_pos, or None."""
        base_hit_radius = 12.0

        for idx, node in enumerate(self.nodes):
            view_pos = self.mapFromScene(node)
            click_view_pos = self.mapFromScene(scene_pos)
            distance = (view_pos - click_view_pos).manhattanLength()
            if distance < base_hit_radius:
                return idx
        return None

    def add_node(self, scene_pos):
        """Add node at scene position."""
        if len(self.nodes) < 4:
            self.nodes.append(scene_pos)
            self.viewport().update()
            # Trigger callback after adding node
            if self.node_change_callback:
                self.node_change_callback()

    def set_node_change_callback(self, callback):
        """Set callback function for node changes."""
        self.node_change_callback = callback

    def set_grid_repetitions(self, h_reps, v_reps):
        """Set grid repetitions and update display."""
        self.grid_h_reps = h_reps
        self.grid_v_reps = v_reps
        self.viewport().update()

    def draw_grid(self, painter):
        """Draw perspective-distorted grid overlay showing rectified cell mapping."""
        if not self.image_item or len(self.nodes) != 4:
            return

        # Only draw grid if we have more than 1 repetition in either direction
        if self.grid_h_reps <= 1 and self.grid_v_reps <= 1:
            return

        try:
            import cv2

            # Convert nodes to numpy array (source quadrilateral in scene coordinates)
            src_points = np.array([[pt.x(), pt.y()]
                                  for pt in self.nodes], dtype=np.float32)

            # Define a reference rectangle for the rectified space
            ref_width = 400
            ref_height = int(ref_width * self.grid_v_reps / self.grid_h_reps)

            dst_points = np.array([
                [0, 0],
                [ref_width, 0],
                [ref_width, ref_height],
                [0, ref_height]
            ], dtype=np.float32)

            # Calculate the inverse perspective transformation (rectified -> original)
            transform_matrix = cv2.getPerspectiveTransform(
                dst_points, src_points)

            # Generate all grid line points first
            vertical_lines = []
            horizontal_lines = []

            # Generate vertical grid lines
            if self.grid_h_reps > 1:
                for i in range(1, self.grid_h_reps):
                    x = i * ref_width / self.grid_h_reps

                    # Create points along this vertical line in rectified space
                    line_points = []
                    # 50 segments per line
                    for y in range(0, ref_height + 1, max(1, ref_height // 50)):
                        rect_point = np.array([[[x, y]]], dtype=np.float32)
                        orig_point = cv2.perspectiveTransform(
                            rect_point, transform_matrix)[0][0]
                        view_point = self.mapFromScene(
                            QPointF(orig_point[0], orig_point[1]))
                        line_points.append(view_point)

                    vertical_lines.append(line_points)

            # Generate horizontal grid lines
            if self.grid_v_reps > 1:
                for i in range(1, self.grid_v_reps):
                    y = i * ref_height / self.grid_v_reps

                    # Create points along this horizontal line in rectified space
                    line_points = []
                    # 50 segments per line
                    for x in range(0, ref_width + 1, max(1, ref_width // 50)):
                        rect_point = np.array([[[x, y]]], dtype=np.float32)
                        orig_point = cv2.perspectiveTransform(
                            rect_point, transform_matrix)[0][0]
                        view_point = self.mapFromScene(
                            QPointF(orig_point[0], orig_point[1]))
                        line_points.append(view_point)

                    horizontal_lines.append(line_points)

            # Draw white background lines first
            white_pen = QPen(QColor(255, 255, 255), 3.0)  # White, thick
            painter.setPen(white_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Draw white background for vertical lines
            for line_points in vertical_lines:
                for j in range(len(line_points) - 1):
                    painter.drawLine(line_points[j], line_points[j + 1])

            # Draw white background for horizontal lines
            for line_points in horizontal_lines:
                for j in range(len(line_points) - 1):
                    painter.drawLine(line_points[j], line_points[j + 1])

            # Draw black grid lines on top
            black_pen = QPen(QColor(0, 0, 0), 1.0)  # Black, thin
            painter.setPen(black_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Draw black vertical lines
            for line_points in vertical_lines:
                for j in range(len(line_points) - 1):
                    painter.drawLine(line_points[j], line_points[j + 1])

            # Draw black horizontal lines
            for line_points in horizontal_lines:
                for j in range(len(line_points) - 1):
                    painter.drawLine(line_points[j], line_points[j + 1])

        except Exception as e:
            # If there's any error with the perspective transformation, fall back to no grid
            pass
