from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QApplication
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QTransform, QPen, QBrush, QColor
import numpy as np
from PIL import Image

class HighPerformanceImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
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
        
    def set_image(self, image_path):
        """Load and display an image from file path."""
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        np_image = np.array(pil_image)
        
        height, width, channel = np_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(np_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.original_image = q_image
        self.display_image(q_image)
        
    def display_image(self, q_image):
        """Display the QImage in the view."""
        if self.image_item:
            self.scene.removeItem(self.image_item)
            
        pixmap = QPixmap.fromImage(q_image)
        self.image_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        
        self.current_zoom = 1.0
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        
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
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
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
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
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
            return
        super().mouseReleaseEvent(event)

    def drawForeground(self, painter, rect):
        """Draw zoom-independent overlays."""
        painter.save()
        painter.resetTransform()
        
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
            color = QColor(255, 140, 0) if len(self.nodes) < 4 else QColor(0, 120, 255)
            colored_pen = QPen(color, base_line_width)
            painter.setPen(colored_pen)
            
            for i in range(len(view_nodes) - 1):
                painter.drawLine(view_nodes[i], view_nodes[i+1])
            if len(view_nodes) == 4:
                painter.drawLine(view_nodes[3], view_nodes[0])

        for idx, node in enumerate(self.nodes):
            view_pos = self.mapFromScene(node)
            self.draw_node(painter, view_pos, base_node_radius, selected=(idx == self.dragging_idx))
        
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