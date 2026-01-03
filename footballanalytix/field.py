"""
Field detection and homography transformation utilities.

Handles soccer pitch configuration, keypoint detection, and 
homography computation for mapping between frame and pitch coordinates.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Keypoint Names (29 keypoints for soccer pitch)
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = {
    1: "sideline_top_left",
    2: "big_rect_left_top_pt1",
    3: "big_rect_left_top_pt2",
    4: "big_rect_left_bottom_pt1",
    5: "big_rect_left_bottom_pt2",
    6: "small_rect_left_top_pt1",
    7: "small_rect_left_top_pt2",
    8: "small_rect_left_bottom_pt1",
    9: "small_rect_left_bottom_pt2",
    10: "sideline_bottom_left",
    11: "left_semicircle_right",
    12: "center_line_top",
    13: "center_line_bottom",
    14: "center_circle_top",
    15: "center_circle_bottom",
    16: "field_center",
    17: "sideline_top_right",
    18: "big_rect_right_top_pt1",
    19: "big_rect_right_top_pt2",
    20: "big_rect_right_bottom_pt1",
    21: "big_rect_right_bottom_pt2",
    22: "small_rect_right_top_pt1",
    23: "small_rect_right_top_pt2",
    24: "small_rect_right_bottom_pt1",
    25: "small_rect_right_bottom_pt2",
    26: "sideline_bottom_right",
    27: "right_semicircle_left",
    28: "center_circle_left",
    29: "center_circle_right",
}


# ---------------------------------------------------------------------------
# Soccer Pitch Configuration
# ---------------------------------------------------------------------------

@dataclass
class SoccerPitchConfiguration:
    """
    Configuration for soccer pitch dimensions and geometry.
    
    All measurements are in the pitch's internal coordinate system
    (default: 12000x7000 units).
    """
    length: int = 12000
    width: int = 7000
    penalty_box_length: int = 1886
    penalty_box_width: int = 4140
    goal_box_length: int = 629
    goal_box_width: int = 1885
    centre_circle_radius: int = 942
    penalty_spot_distance: int = 1257

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        """Get all 29 pitch keypoint vertices."""
        top_penalty = (self.width - self.penalty_box_width) / 2
        bottom_penalty = self.width - top_penalty
        top_goal = (self.width - self.goal_box_width) / 2
        bottom_goal = self.width - top_goal
        center_y = self.width / 2
        arc_radius = self.centre_circle_radius
        penalty_spot = self.penalty_spot_distance
        
        return [
            (0, 0),  # 1 sideline_top_left
            (0, top_penalty),  # 2 big_rect_left_top_pt1
            (self.penalty_box_length, top_penalty),  # 3 big_rect_left_top_pt2
            (0, bottom_penalty),  # 4 big_rect_left_bottom_pt1
            (self.penalty_box_length, bottom_penalty),  # 5 big_rect_left_bottom_pt2
            (0, top_goal),  # 6 small_rect_left_top_pt1
            (self.goal_box_length, top_goal),  # 7 small_rect_left_top_pt2
            (0, bottom_goal),  # 8 small_rect_left_bottom_pt1
            (self.goal_box_length, bottom_goal),  # 9 small_rect_left_bottom_pt2
            (0, self.width),  # 10 sideline_bottom_left
            (penalty_spot + arc_radius, center_y),  # 11 left_semicircle_right
            (self.length / 2, 0),  # 12 center_line_top
            (self.length / 2, self.width),  # 13 center_line_bottom
            (self.length / 2, center_y - arc_radius),  # 14 center_circle_top
            (self.length / 2, center_y + arc_radius),  # 15 center_circle_bottom
            (self.length / 2, center_y),  # 16 field_center
            (self.length, 0),  # 17 sideline_top_right
            (self.length, top_penalty),  # 18 big_rect_right_top_pt1
            (self.length - self.penalty_box_length, top_penalty),  # 19 big_rect_right_top_pt2
            (self.length, bottom_penalty),  # 20 big_rect_right_bottom_pt1
            (self.length - self.penalty_box_length, bottom_penalty),  # 21 big_rect_right_bottom_pt2
            (self.length, top_goal),  # 22 small_rect_right_top_pt1
            (self.length - self.goal_box_length, top_goal),  # 23 small_rect_right_top_pt2
            (self.length, bottom_goal),  # 24 small_rect_right_bottom_pt1
            (self.length - self.goal_box_length, bottom_goal),  # 25 small_rect_right_bottom_pt2
            (self.length, self.width),  # 26 sideline_bottom_right
            (self.length - (penalty_spot + arc_radius), center_y),  # 27 right_semicircle_left
            (self.length / 2 - arc_radius, center_y),  # 28 center_circle_left
            (self.length / 2 + arc_radius, center_y),  # 29 center_circle_right
        ]

    edges: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            # Boundary
            (1, 17), (1, 10), (17, 26), (10, 26),
            # Left penalty box
            (2, 3), (4, 5), (2, 4), (3, 5),
            # Left goal box
            (6, 7), (8, 9), (6, 8), (7, 9),
            # Right penalty box
            (18, 19), (20, 21), (18, 20), (19, 21),
            # Right goal box
            (22, 23), (24, 25), (22, 24), (23, 25),
            # Center line
            (12, 13),
        ]
    )

    line_edges: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            # Boundary
            (1, 10), (1, 17), (17, 26), (10, 26),
            # Left penalty box
            (2, 3), (2, 4), (3, 5), (4, 5),
            # Left goal box
            (6, 7), (6, 8), (7, 9), (8, 9),
            # Right penalty box
            (18, 19), (18, 20), (19, 21), (20, 21),
            # Right goal box
            (22, 23), (22, 24), (23, 25), (24, 25),
            # Center line and circle spokes
            (12, 14), (13, 15), (14, 16), (15, 16),
            (14, 28), (14, 29), (15, 28), (15, 29), (28, 29)
        ]
    )


# ---------------------------------------------------------------------------
# View Transformer (Homography)
# ---------------------------------------------------------------------------

class ViewTransformer:
    """
    Handles perspective transformation between coordinate systems.
    
    Uses homography to transform points between video frame coordinates
    and pitch/minimap coordinates.
    """
    
    def __init__(
        self,
        source: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32]
    ) -> None:
        """
        Initialize transformer with source and target point correspondences.
        
        Args:
            source: Nx2 array of source points
            target: Nx2 array of target points
        
        Raises:
            ValueError: If shapes don't match or homography fails
        """
        if source.shape != target.shape or source.shape[1] != 2:
            raise ValueError("Source and target must be Nx2 arrays with matching shapes")
        
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        
        self.m, _ = cv2.findHomography(
            source, target, 
            method=cv2.USAC_MAGSAC, 
            ransacReprojThreshold=3.0
        )
        
        if self.m is None:
            raise ValueError("Homography matrix could not be computed.")

    def transform_points(
        self,
        points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Transform points using the homography matrix.
        
        Args:
            points: Nx2 array of points to transform
        
        Returns:
            Nx2 array of transformed points
        """
        if points.size == 0:
            return points
        
        if points.shape[1] != 2:
            raise ValueError("Points must be Nx2 coordinates")
        
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2).astype(np.float32)

    def transform_image(
        self,
        image: np.ndarray,
        resolution_wh: Tuple[int, int]
    ) -> np.ndarray:
        """
        Warp an image using the homography matrix.
        
        Args:
            image: Input image (grayscale or color)
            resolution_wh: Output resolution (width, height)
        
        Returns:
            Warped image
        """
        if len(image.shape) not in {2, 3}:
            raise ValueError("Image must be grayscale or color")
        return cv2.warpPerspective(image, self.m, resolution_wh)


# ---------------------------------------------------------------------------
# Field Detection Functions
# ---------------------------------------------------------------------------

def detect_field_keypoints(
    frame: np.ndarray,
    model,
    config: SoccerPitchConfiguration,
    confidence: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect field keypoints and return matched pitch/frame point pairs.
    
    Args:
        frame: Video frame (BGR)
        model: YOLO keypoint detection model
        config: Soccer pitch configuration
        confidence: Detection confidence threshold
    
    Returns:
        Tuple of (frame_points, pitch_points) as Nx2 arrays
    """
    result = model(frame, conf=confidence, verbose=False)[0]
    keypoints = getattr(result, "keypoints", None)
    
    if keypoints is None or keypoints.xy is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    xy = keypoints.xy
    conf_arr = getattr(keypoints, "conf", None)

    xy_np = xy[0].cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)[0]
    conf_np = None
    if conf_arr is not None:
        conf_np = conf_arr[0].cpu().numpy() if hasattr(conf_arr, "cpu") else np.asarray(conf_arr)[0]

    expected = len(config.vertices)
    xy_np = xy_np[:expected]
    if conf_np is not None:
        conf_np = conf_np[:expected]

    if xy_np.size == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    mask = conf_np > confidence if conf_np is not None else np.ones(len(xy_np), dtype=bool)
    if not mask.any():
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    pitch_vertices = np.array(config.vertices, dtype=np.float32)[: len(xy_np)]
    frame_points = xy_np[mask].astype(np.float32)
    pitch_points = pitch_vertices[mask]
    
    return frame_points, pitch_points


def compute_view_transformers(
    frame: np.ndarray,
    model,
    config: SoccerPitchConfiguration,
    confidence: float = 0.65,
) -> Tuple[Optional[ViewTransformer], Optional[ViewTransformer], np.ndarray, np.ndarray]:
    """
    Compute forward and inverse view transformers from detected keypoints.
    
    Args:
        frame: Video frame (BGR)
        model: YOLO keypoint detection model
        config: Soccer pitch configuration
        confidence: Detection confidence threshold
    
    Returns:
        Tuple of (pitch_to_frame, frame_to_pitch, frame_points, pitch_points)
        Transformers are None if not enough keypoints detected
    """
    frame_points, pitch_points = detect_field_keypoints(frame, model, config, confidence)
    
    if len(frame_points) < 4 or len(pitch_points) < 4:
        return None, None, frame_points, pitch_points
    
    forward = ViewTransformer(source=pitch_points, target=frame_points)
    inverse = ViewTransformer(source=frame_points, target=pitch_points)
    
    return forward, inverse, frame_points, pitch_points
