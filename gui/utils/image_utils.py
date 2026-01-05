"""
PyQt6 Image format utilities.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def cv2_to_qimage(cv2_image: np.ndarray) -> QImage:
    """
    Convert OpenCV BGR image to QImage.
    
    Args:
        cv2_image: OpenCV image in BGR format
        
    Returns:
        QImage in RGB888 format
    """
    height, width = cv2_image.shape[:2]
    bytes_per_line = 3 * width
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Create QImage from data
    # Note: We must keep a reference to the data if we don't copy, 
    # but QImage(data, ...) creates a view. QImage(...).copy() makes it safe.
    q_image = QImage(
        rgb_image.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888
    )
    
    return q_image.copy()  # Copy to decouple from numpy array


def cv2_to_pixmap(cv2_image: np.ndarray) -> QPixmap:
    """
    Convert OpenCV image directly to QPixmap for display.
    """
    return QPixmap.fromImage(cv2_to_qimage(cv2_image))


def resize_keeping_aspect_ratio(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """
    Resize image using OpenCV (faster than Qt scaling usually).
    """
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    
    if scale >= 1.0:
        return image
        
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
