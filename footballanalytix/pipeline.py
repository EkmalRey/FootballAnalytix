"""
Main processing pipeline for FootballAnalytix.

Combines player detection, field detection, team clustering, and 
visualization into cohesive processing functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast, TYPE_CHECKING

import cv2
import numpy as np
import supervision as sv

if TYPE_CHECKING:
    from ultralytics import YOLO

from .config import (
    CONFIDENCE_THRESHOLD,
    REASSIGN_INTERVAL,
    MAX_COLOR_SAMPLES,
    TEAM_COLORS_BGR,
    UNASSIGNED_COLOR,
    MINIMAP_KEYPOINT_COLOR,
    DEFAULT_VIDEO_PATH,
    CLEANED_DATASET_DIR,
)
from .state import CombinedState
from .models import get_model_label
from .players import (
    extract_jersey_color,
    collect_initial_team_colors,
    assign_player_team,
    reassign_all_teams,
    resolve_class_metadata,
    maybe_initialize_team_colors,
)
from .field import (
    SoccerPitchConfiguration,
    compute_view_transformers,
    validate_homography,
    check_keypoint_distribution,
)
from .visualization import (
    draw_pitch,
    minimap_coords,
    create_annotators,
)


# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------

def initialize_combined_state(
    video_path: Path,
    model: YOLO,
    num_frames: int = 30,
    conf: float = CONFIDENCE_THRESHOLD,
) -> CombinedState:
    """
    Initialize combined processing state from a video.
    
    Collects initial team colors from random frames and creates
    a state object for tracking.
    
    Args:
        video_path: Path to video file
        model: YOLO player detection model
        num_frames: Number of frames to sample for color initialization
        conf: Detection confidence threshold
    
    Returns:
        Initialized CombinedState object
    
    Raises:
        FileNotFoundError: If video file doesn't exist
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video at {video_path}")
    
    try:
        team_centers, samples = collect_initial_team_colors(
            cap, model, num_frames=num_frames, conf=conf, return_samples=True
        )
    finally:
        cap.release()
    
    state = CombinedState(team_color_centers=team_centers, color_samples=list(samples))
    
    if len(state.color_samples) > MAX_COLOR_SAMPLES:
        state.color_samples = state.color_samples[-MAX_COLOR_SAMPLES:]
    
    if state.team_color_centers is None:
        print("Team colors will be initialized once enough samples are collected during playback.")
    
    return state


def create_state_from_frame(
    frame: np.ndarray,
    model: YOLO,
    conf: float = CONFIDENCE_THRESHOLD,
    image_path: Optional[Path] = None,
) -> CombinedState:
    """
    Create processing state from a single frame.
    
    Useful for single-image processing or previews.
    
    Args:
        frame: BGR image
        model: YOLO player detection model
        conf: Detection confidence threshold
        image_path: Optional path for model input (uses frame if None)
    
    Returns:
        Initialized CombinedState object
    """
    from sklearn.cluster import KMeans
    
    predict_source = str(image_path) if image_path is not None else frame
    results = model.predict(source=predict_source, conf=conf, verbose=False)
    boxes = results[0].boxes
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    jersey_colors: List[np.ndarray] = []

    if len(boxes) == 0:
        print("No detections found to derive team colors; continuing without initialization.")
    
    for box in boxes:
        # Player class check (class_id == 1)
        if int(box.cls[0]) != 1:
            continue
        color = extract_jersey_color(frame_rgb, box.xyxy[0].cpu().numpy())
        if color is not None:
            jersey_colors.append(color)

    centers: Optional[np.ndarray] = None
    if len(jersey_colors) >= 2:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(np.array(jersey_colors))
        centers = kmeans.cluster_centers_
        print("Initialized team colors from single frame preview.")
    elif jersey_colors:
        print("Only one player sample available; waiting for more detections to cluster teams.")
    else:
        print("No player samples extracted; state will bootstrap later.")

    state = CombinedState(team_color_centers=centers, color_samples=list(jersey_colors))
    
    if len(state.color_samples) > MAX_COLOR_SAMPLES:
        state.color_samples = state.color_samples[-MAX_COLOR_SAMPLES:]
    
    return state


def try_initialize_team_colors(state: CombinedState) -> bool:
    """
    Try to initialize team colors and reassign tracked players.
    
    Args:
        state: Combined processing state
    
    Returns:
        True if colors were just initialized, False otherwise
    """
    new_centers = maybe_initialize_team_colors(
        state.team_color_centers, 
        state.color_samples
    )
    
    if new_centers is not None and state.team_color_centers is None:
        state.team_color_centers = new_centers
        if state.player_colors_cache:
            state.player_teams = reassign_all_teams(
                state.player_teams, 
                state.player_colors_cache, 
                state.team_color_centers
            )
        return True
    
    return False


# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------

def process_combined_frame(
    frame: np.ndarray,
    frame_idx: int,
    state: CombinedState,
    players_model: YOLO,
    field_model: YOLO,
    config: Optional[SoccerPitchConfiguration] = None,
    conf: float = CONFIDENCE_THRESHOLD,
    show_edges: bool = False,
    show_vertices: bool = False,
    show_detected_vertices: bool = True,
    show_object_detection: bool = True,
    show_clustering: bool = True,
    show_id: bool = True,
    base_minimap: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Process a single frame with player and field detection.
    
    This is the main processing function that:
    1. Updates team colors if needed
    2. Computes homography from field keypoints
    3. Detects and tracks players
    4. Assigns players to teams
    5. Projects positions to minimap
    6. Draws annotations
    
    Args:
        frame: BGR video frame
        frame_idx: Frame index (for periodic reassignment)
        state: Processing state object
        players_model: YOLO player detection model
        field_model: YOLO field keypoint model
        config: Pitch configuration (uses default if None)
        conf: Detection confidence threshold
        show_edges: Draw pitch edges on frame
        show_vertices: Draw pitch vertices on frame
        show_detected_vertices: Draw detected keypoints on frame
        show_object_detection: Draw bounding boxes
        show_clustering: Use team clustering for colors
        show_id: Show track IDs on bounding boxes
        base_minimap: Pre-rendered minimap (computed if None)
    
    Returns:
        Dictionary containing:
        - 'frame': Annotated frame
        - 'minimap': Minimap with projected positions
        - 'stats': Detection statistics
        - 'pitch_points': Player positions on pitch
        - 'class_styles': Class style information
    """
    if config is None:
        config = SoccerPitchConfiguration()
    
    # Try to initialize team colors
    try_initialize_team_colors(state)

    # Get frame dimensions for keypoint distribution check
    frame_height = frame.shape[0]
    
    # Always run detection every frame (smooth tracking)
    pitch_to_frame, frame_to_pitch, frame_pts, pitch_pts, avg_conf = compute_view_transformers(
        frame, field_model, config, conf
    )
    
    # Validate the new homography before accepting it
    should_update = False
    if frame_to_pitch is not None and pitch_to_frame is not None:
        # Check 1: Keypoint distribution (prevents collinear point issues)
        has_good_distribution = check_keypoint_distribution(
            frame_pts, frame_height, min_vertical_ratio=0.10
        )
        
        # Check 2: Minimum keypoints for stability
        has_enough_keypoints = len(frame_pts) >= 6
        
        # Check 3: Reprojection error validation
        is_valid, reproj_error = validate_homography(
            frame_to_pitch, frame_pts, pitch_pts
        )
        
        # Accept if all checks pass, OR if we have no existing homography
        if not state.has_homography():
            # First valid homography - accept with minimal requirements
            should_update = len(frame_pts) >= 4 and is_valid
        else:
            # Already have one - be more strict
            should_update = has_good_distribution and has_enough_keypoints and is_valid
    
    if should_update:
        state.pitch_to_frame = pitch_to_frame
        state.frame_to_pitch = frame_to_pitch
        state.last_frame_points = frame_pts
        state.last_pitch_points = pitch_pts
        state.last_homography_confidence = avg_conf
        state.homography_frame_idx = frame_idx

    # Project detected keypoints for visualization
    detected_pitch_points = np.empty((0, 2), dtype=np.float32)
    if state.frame_to_pitch is not None and frame_pts.size:
        detected_pitch_points = state.frame_to_pitch.transform_points(
            frame_pts.astype(np.float32)
        )

    # Periodic team reassignment
    if (
        state.team_color_centers is not None
        and frame_idx % REASSIGN_INTERVAL == 0
        and state.player_colors_cache
    ):
        centers = cast(np.ndarray, state.team_color_centers)
        state.player_teams = reassign_all_teams(
            state.player_teams, state.player_colors_cache, centers
        )

    # Run player tracking
    results = players_model.track(source=frame, conf=conf, persist=True, verbose=False)
    boxes = results[0].boxes
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_frame = frame.copy()

    # Statistics
    team_counts = {0: 0, 1: 0}
    unassigned_count = 0
    other_count = 0
    class_breakdown: Dict[str, int] = {}
    class_styles: Dict[str, Dict[str, Any]] = {}
    player_ground_points: List[Tuple[float, float]] = []
    player_team_labels: List[Optional[int]] = []
    other_ground_points: List[Tuple[Tuple[float, float], Tuple[int, int, int], int]] = []

    # Process detections
    if boxes.id is not None or len(boxes) > 0:
        for det_idx, box in enumerate(boxes):
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else frame_idx * 1000 + det_idx

            legend_name, class_color, is_player, show_on_minimap, minimap_radius, _ = resolve_class_metadata(
                cls, players_model
            )
            class_breakdown[legend_name] = class_breakdown.get(legend_name, 0) + 1
            class_styles[legend_name] = {"color": class_color, "is_player": is_player}

            if is_player:
                # Extract jersey color
                jersey_color = extract_jersey_color(frame_rgb, box.xyxy[0].cpu().numpy())
                if jersey_color is not None:
                    state.player_colors_cache[track_id] = jersey_color
                    state.color_samples.append(jersey_color)
                    if len(state.color_samples) > MAX_COLOR_SAMPLES:
                        state.color_samples.pop(0)
                    try_initialize_team_colors(state)

                # Assign team
                centers_opt = state.team_color_centers if show_clustering else None
                team_label: Optional[int] = None
                
                if centers_opt is not None:
                    color_source = jersey_color if jersey_color is not None else state.player_colors_cache.get(track_id)
                    if color_source is not None:
                        centers = cast(np.ndarray, centers_opt)
                        team_label = assign_player_team(color_source, centers)
                        state.player_teams[track_id] = team_label
                    else:
                        team_label = state.player_teams.get(track_id)

                # Determine display color
                if team_label is None:
                    unassigned_count += 1
                    box_color = UNASSIGNED_COLOR
                    label_prefix = "U"
                else:
                    team_counts[team_label] += 1
                    box_color = TEAM_COLORS_BGR[team_label]
                    label_prefix = f"T{team_label}"

                # Draw bounding box
                if show_object_detection:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    if show_id:
                        cv2.putText(
                            annotated_frame,
                            f"{label_prefix}-{track_id}",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            box_color,
                            2,
                        )

                # Store ground point (foot position)
                foot_x = (x1 + x2) / 2
                foot_y = y2
                player_ground_points.append((foot_x, foot_y))
                player_team_labels.append(team_label)
            else:
                # Non-player detection
                other_count += 1
                default_label = get_model_label(players_model, cls)
                display_label = legend_name or default_label
                
                if show_object_detection:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_color, 2)
                    if show_id:
                        cv2.putText(
                            annotated_frame,
                            f"{display_label}-{track_id}",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            class_color,
                            2,
                        )
                
                if show_on_minimap:
                    ground_point = ((x1 + x2) / 2, y2)
                    other_ground_points.append((ground_point, class_color, minimap_radius))

    # Create minimap
    if base_minimap is not None:
        minimap = base_minimap.copy()
    else:
        minimap = draw_pitch(config)
    
    # Project players to minimap
    pitch_player_points = np.empty((0, 2), dtype=np.float32)
    if state.frame_to_pitch is not None and player_ground_points:
        pitch_player_points = state.frame_to_pitch.transform_points(
            np.array(player_ground_points, dtype=np.float32)
        )
        for (x, y), team in zip(pitch_player_points, player_team_labels):
            color = TEAM_COLORS_BGR[team] if team is not None else UNASSIGNED_COLOR
            cv2.circle(
                minimap,
                minimap_coords(x, y),
                12,
                color,
                -1,
            )

    # Project other objects to minimap
    if state.frame_to_pitch is not None and other_ground_points:
        other_points = np.array([pt for pt, _, _ in other_ground_points], dtype=np.float32)
        projected_other = state.frame_to_pitch.transform_points(other_points)
        for (x, y), (_, color, radius) in zip(projected_other, other_ground_points):
            cv2.circle(
                minimap,
                minimap_coords(x, y),
                radius,
                color,
                -1,
            )

    # Draw keypoints on minimap
    if state.last_pitch_points.size:
        for px, py in state.last_pitch_points:
            cv2.circle(
                minimap,
                minimap_coords(px, py),
                6,
                MINIMAP_KEYPOINT_COLOR,
                2,
            )

    if detected_pitch_points.size:
        for px, py in detected_pitch_points:
            cv2.circle(
                minimap,
                minimap_coords(px, py),
                6,
                (0, 0, 0),
                2,
            )

    # Draw detected vertices on frame
    if state.pitch_to_frame is not None and show_detected_vertices and state.last_frame_points.size:
        _, _, detected_vertex_annotator = create_annotators(config)
        ref_points = sv.KeyPoints(xy=state.last_frame_points[np.newaxis, ...])
        annotated_frame = detected_vertex_annotator.annotate(scene=annotated_frame, key_points=ref_points)
        
        vertices_array = np.array(config.vertices, dtype=np.float32)
        for (fx, fy), pitch_pt in zip(state.last_frame_points, state.last_pitch_points):
            idx = int(np.argmin(np.linalg.norm(vertices_array - pitch_pt, axis=1)))
            label_id = idx + 1
            cv2.putText(
                annotated_frame,
                str(label_id),
                (int(fx) + 4, max(0, int(fy) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                MINIMAP_KEYPOINT_COLOR,
                2,
            )

    # Compile statistics
    stats = {
        "team_0": team_counts[0],
        "team_1": team_counts[1],
        "unassigned": unassigned_count,
        "other": other_count,
        "tracked": len(state.player_teams),
        "classes": class_breakdown,
    }

    return {
        "frame": annotated_frame,
        "minimap": minimap,
        "stats": stats,
        "pitch_points": pitch_player_points,
        "class_styles": class_styles,
    }
