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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Detect field keypoints and return matched pitch/frame point pairs.
    
    Args:
        frame: Video frame (BGR)
        model: YOLO keypoint detection model
        config: Soccer pitch configuration
        confidence: Detection confidence threshold
    
    Returns:
        Tuple of (frame_points, pitch_points, avg_confidence) as Nx2 arrays and float
    """
    result = model(frame, conf=confidence, verbose=False)[0]
    keypoints = getattr(result, "keypoints", None)
    
    if keypoints is None or keypoints.xy is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            0.0,
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
            0.0,
        )

    mask = conf_np > confidence if conf_np is not None else np.ones(len(xy_np), dtype=bool)
    if not mask.any():
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            0.0,
        )

    pitch_vertices = np.array(config.vertices, dtype=np.float32)[: len(xy_np)]
    frame_points = xy_np[mask].astype(np.float32)
    pitch_points = pitch_vertices[mask]
    
    # Compute average confidence of valid keypoints
    avg_confidence = float(np.mean(conf_np[mask])) if conf_np is not None else 1.0
    
    return frame_points, pitch_points, avg_confidence


def compute_view_transformers(
    frame: np.ndarray,
    model,
    config: SoccerPitchConfiguration,
    confidence: float = 0.65,
) -> Tuple[Optional[ViewTransformer], Optional[ViewTransformer], np.ndarray, np.ndarray, float]:
    """
    Compute forward and inverse view transformers from detected keypoints.
    
    Args:
        frame: Video frame (BGR)
        model: YOLO keypoint detection model
        config: Soccer pitch configuration
        confidence: Detection confidence threshold
    
    Returns:
        Tuple of (pitch_to_frame, frame_to_pitch, frame_points, pitch_points, avg_confidence)
        Transformers are None if not enough keypoints detected
    """
    frame_points, pitch_points, avg_conf = detect_field_keypoints(frame, model, config, confidence)
    
    if len(frame_points) < 4 or len(pitch_points) < 4:
        return None, None, frame_points, pitch_points, avg_conf
    
    forward = ViewTransformer(source=pitch_points, target=frame_points)
    inverse = ViewTransformer(source=frame_points, target=pitch_points)
    
    return forward, inverse, frame_points, pitch_points, avg_conf


# ---------------------------------------------------------------------------
# Homography Stabilization
# ---------------------------------------------------------------------------

# Configuration constants
HOMOGRAPHY_INTERVAL = 3       # Frames between detection attempts
MIN_KEYPOINTS_STABLE = 6      # Minimum keypoints for stable homography
MAX_REPROJ_ERROR = 150.0      # Maximum allowed reprojection error (pitch units)
CONFIDENCE_THRESHOLD_MULT = 0.85  # Only update if new confidence > old * this


def validate_homography(
    transformer: ViewTransformer,
    frame_points: npt.NDArray[np.float32],
    pitch_points: npt.NDArray[np.float32],
    max_error: float = MAX_REPROJ_ERROR,
) -> Tuple[bool, float]:
    """
    Validate a homography by checking reprojection error.
    
    Args:
        transformer: The ViewTransformer to validate
        frame_points: Detected frame coordinates (Nx2)
        pitch_points: Corresponding pitch coordinates (Nx2)
        max_error: Maximum allowed mean reprojection error
    
    Returns:
        Tuple of (is_valid, mean_error)
    """
    if frame_points.size == 0 or pitch_points.size == 0:
        return False, float('inf')
    
    # Check matrix is finite
    if not np.isfinite(transformer.m).all():
        return False, float('inf')
    
    # Transform frame points to pitch and measure error
    try:
        projected = transformer.transform_points(frame_points)
        errors = np.linalg.norm(projected - pitch_points, axis=1)
        mean_error = float(np.mean(errors))
        
        return mean_error < max_error, mean_error
    except Exception:
        return False, float('inf')


def check_keypoint_distribution(
    frame_points: npt.NDArray[np.float32],
    frame_height: int,
    min_vertical_ratio: float = 0.15,
) -> bool:
    """
    Check if keypoints are distributed well enough for stable homography.
    
    Rejects sets where all points are on a single horizontal band,
    which causes unstable perspective transforms.
    
    Args:
        frame_points: Detected frame coordinates (Nx2)
        frame_height: Height of the video frame
        min_vertical_ratio: Minimum vertical spread as fraction of frame height
    
    Returns:
        True if distribution is acceptable
    """
    if frame_points.size == 0:
        return False
    
    y_coords = frame_points[:, 1]
    y_spread = np.max(y_coords) - np.min(y_coords)
    
    return (y_spread / frame_height) >= min_vertical_ratio


def should_update_homography(
    frame_idx: int,
    current_confidence: float,
    last_confidence: float,
    last_update_frame: int,
    has_existing: bool,
    num_keypoints: int,
    is_valid: bool,
    force_interval: int = HOMOGRAPHY_INTERVAL,
) -> bool:
    """
    Decide whether to update the homography transformer.
    
    Uses a hybrid strategy combining:
    - Frame interval skipping (performance)
    - Keypoint count requirements (quality)
    - Confidence comparison (stability)
    - Validity checking (correctness)
    
    Args:
        frame_idx: Current frame index
        current_confidence: Average confidence of current detections
        last_confidence: Confidence when homography was last updated
        last_update_frame: Frame index when homography was last updated
        has_existing: Whether a valid homography already exists
        num_keypoints: Number of keypoints detected
        is_valid: Whether the new homography passed validation
        force_interval: Frames between forced update attempts
    
    Returns:
        True if homography should be updated
    """
    # Case 1: No existing homography - must accept if valid
    if not has_existing:
        return is_valid and num_keypoints >= 4
    
    # Case 2: Not enough keypoints for stable result
    if num_keypoints < MIN_KEYPOINTS_STABLE:
        return False
    
    # Case 3: New homography failed validation
    if not is_valid:
        return False
    
    # Case 4: Frame interval check (skip frames for performance)
    frames_since_update = frame_idx - last_update_frame
    if frames_since_update < force_interval:
        return False
    
    # Case 5: Confidence comparison - only update if quality is similar or better
    if current_confidence < last_confidence * CONFIDENCE_THRESHOLD_MULT:
        return False
    
    return True


def compute_keypoint_confidence(
    keypoints_conf: Optional[npt.NDArray[np.float32]],
    mask: npt.NDArray[np.bool_],
) -> float:
    """
    Compute average confidence of valid keypoints.
    
    Args:
        keypoints_conf: Confidence array from model (may be None)
        mask: Boolean mask of valid keypoints
    
    Returns:
        Average confidence (0.0 if no valid keypoints)
    """
    if keypoints_conf is None or not mask.any():
        return 0.0
    
    return float(np.mean(keypoints_conf[mask]))
