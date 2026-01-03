"""
Player detection and team clustering utilities.

Handles jersey color extraction, team assignment via KMeans clustering,
and class metadata resolution.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Union, Any, cast, TYPE_CHECKING

import cv2
import numpy as np
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from ultralytics import YOLO

from .config import (
    CONFIDENCE_THRESHOLD,
    MAX_COLOR_SAMPLES,
    CLASS_CONFIG,
    CLASS_ID_LOOKUP,
    DEFAULT_OTHER_COLOR,
    DEFAULT_MINIMAP_RADIUS,
)
from .models import get_model_label


def extract_jersey_color(
    frame_rgb: np.ndarray,
    box_xyxy: np.ndarray
) -> Optional[np.ndarray]:
    """
    Extract mean RGB color from the upper half of a player's bounding box.
    
    The upper half typically contains the jersey, making it more reliable
    for team color extraction.
    
    Args:
        frame_rgb: Frame in RGB format
        box_xyxy: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        Mean RGB color as numpy array, or None if extraction fails.
    """
    x1, y1, x2, y2 = map(int, box_xyxy)
    player_crop = frame_rgb[y1:y2, x1:x2]
    
    if player_crop.size == 0:
        return None
    
    upper_half = player_crop[: max(1, player_crop.shape[0] // 2), :]
    
    if upper_half.size == 0:
        return None
    
    return upper_half.reshape(-1, 3).mean(axis=0)


def collect_initial_team_colors(
    cap: cv2.VideoCapture,
    model: YOLO,
    num_frames: int = 30,
    conf: float = CONFIDENCE_THRESHOLD,
    return_samples: bool = False,
) -> Union[np.ndarray, Tuple[Optional[np.ndarray], List[np.ndarray]]]:
    """
    Collect initial jersey colors from random video frames to bootstrap KMeans.
    
    Samples colors from detected players across multiple random frames
    to establish initial team color centers.
    
    Args:
        cap: OpenCV VideoCapture object
        model: YOLO player detection model
        num_frames: Number of random frames to sample
        conf: Detection confidence threshold
        return_samples: If True, also return the raw color samples
    
    Returns:
        If return_samples=False: numpy array of team color centers
        If return_samples=True: tuple of (centers, samples list)
    
    Raises:
        RuntimeError: If not enough player samples collected (when return_samples=False)
    """
    initial_colors: List[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or num_frames
    target_frames = min(num_frames, total_frames)

    # Select random frame indices
    random_indices = random.sample(range(total_frames), target_frames)
    random_indices.sort()  # Sort to minimize seeking

    print(f"Collecting jersey samples from {target_frames} random frames...")

    for idx in random_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        results = model.predict(source=frame, conf=conf, verbose=False)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for box in results[0].boxes:
            # Check if player class (class_id == 1 in default config)
            if int(box.cls[0]) != 1:  # Player class
                continue
            color = extract_jersey_color(frame_rgb, box.xyxy[0].cpu().numpy())
            if color is not None:
                initial_colors.append(color)

    if len(initial_colors) < 2:
        warning = (
            f"Only collected {len(initial_colors)} player samples; "
            "continuing without initializing team colors."
        )
        print(warning)
        if return_samples:
            return None, initial_colors
        raise RuntimeError("Not enough player samples to establish team colors.")

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(np.array(initial_colors))
    centers = kmeans.cluster_centers_
    print("Team color centers (RGB):\n", centers)
    
    if return_samples:
        return centers, initial_colors
    return centers


def assign_player_team(
    jersey_color: np.ndarray,
    team_color_centers: np.ndarray
) -> int:
    """
    Assign a player to a team based on jersey color distance.
    
    Args:
        jersey_color: RGB color of player's jersey
        team_color_centers: Array of team color cluster centers
    
    Returns:
        Team index (0 or 1)
    """
    dist = np.linalg.norm(team_color_centers - jersey_color, axis=1)
    return int(np.argmin(dist))


def reassign_all_teams(
    player_teams: Dict[int, int],
    current_colors: Dict[int, np.ndarray],
    team_color_centers: np.ndarray,
) -> Dict[int, int]:
    """
    Reassign all tracked players to teams based on current colors.
    
    Used periodically to correct any drift in team assignments.
    
    Args:
        player_teams: Current team assignments
        current_colors: Cached jersey colors per track ID
        team_color_centers: Team color cluster centers
    
    Returns:
        Updated team assignments dictionary
    """
    switches = 0
    updated: Dict[int, int] = {}
    
    for track_id, color in current_colors.items():
        new_team = assign_player_team(color, team_color_centers)
        if player_teams.get(track_id, new_team) != new_team:
            switches += 1
        updated[track_id] = new_team
    
    if switches:
        print(f"Reassigned {switches} players to stabilize teams")
    
    return updated


def resolve_class_metadata(
    class_id: int,
    model: Any,
) -> Tuple[str, Tuple[int, int, int], bool, bool, int, Dict[str, Any]]:
    """
    Map a class ID to configured metadata with sensible fallbacks.
    
    Args:
        class_id: Detection class ID
        model: YOLO model for label lookup
    
    Returns:
        Tuple of (legend_name, color, is_player, show_on_minimap, minimap_radius, config_dict)
    """
    class_name = CLASS_ID_LOOKUP.get(class_id)
    class_config = CLASS_CONFIG.get(class_name, {}) if class_name else {}
    
    legend_name = class_config.get("legend") or (
        class_name.title() if class_name else get_model_label(model, class_id)
    )
    class_color = tuple(class_config.get("color", DEFAULT_OTHER_COLOR))
    is_player = bool(class_config.get("is_player", False))
    show_on_minimap = bool(class_config.get("show_on_minimap", True))
    minimap_radius = int(class_config.get("minimap_radius", DEFAULT_MINIMAP_RADIUS))
    
    return (
        legend_name,
        class_color,
        is_player,
        show_on_minimap,
        minimap_radius,
        class_config,
    )


def maybe_initialize_team_colors(
    team_color_centers: Optional[np.ndarray],
    color_samples: List[np.ndarray]
) -> Optional[np.ndarray]:
    """
    Initialize team color centers once at least two samples exist.
    
    Args:
        team_color_centers: Current centers (may be None)
        color_samples: List of collected color samples
    
    Returns:
        New team color centers, or None if not enough samples
    """
    if team_color_centers is not None:
        return team_color_centers
    
    if len(color_samples) < 2:
        return None
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(np.array(color_samples))
    centers = kmeans.cluster_centers_
    
    print(f"Initialized team colors using {len(color_samples)} accumulated samples.")
    return centers
