"""
Custom Video Widget for high-performance playback.
"""

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont

class VideoDisplayWidget(QLabel):
    """
    Optimized widget for displaying video frames.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        self.setText("No Signal")
        
        self._pixmap = None

    def set_frame(self, pixmap: QPixmap):
        """Update the displayed frame."""
        self._pixmap = pixmap
        # Trigger repaint
        self.update()

    def paintEvent(self, event):
        """Custom paint event for Aspect Ratio handling."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Draw background
        painter.fillRect(self.rect(), QColor("#000000"))
        
        if self._pixmap and not self._pixmap.isNull():
            # Calculate aspect ratio scaled rect
            target_rect = self.rect()
            scaled_pixmap = self._pixmap.scaled(
                target_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Center the image
            x = (target_rect.width() - scaled_pixmap.width()) // 2
            y = (target_rect.height() - scaled_pixmap.height()) // 2
            
            painter.drawPixmap(x, y, scaled_pixmap)
        else:
            # Draw placeholder text
            painter.setPen(QColor("#666666"))
            painter.setFont(QFont("Segoe UI", 16))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Signal")
