"""
FootballAnalytix - Modular Football/Futsal Video Analysis Pipeline

A Python library for analyzing football/futsal videos using computer vision.
Provides player detection, team classification, field detection, and 
2D minimap visualization.

Example usage:
    from footballanalytix import (
        load_players_model,
        load_field_model,
        CombinedState,
        SoccerPitchConfiguration,
        process_combined_frame,
        draw_pitch,
    )
    
    # Load models
    players_model = load_players_model()
    field_model = load_field_model()
    
    # Initialize state
    state = CombinedState()
    config = SoccerPitchConfiguration()
    
    # Process a frame
    result = process_combined_frame(
        frame=image,
        frame_idx=0,
        state=state,
        players_model=players_model,
        field_model=field_model,
        config=config,
    )
"""

__version__ = "1.0.0"
__author__ = "FootballAnalytix Team"

# Configuration and constants
from .config import (
    PROJECT_ROOT,
    MODELS_DIR,
    DATA_DIR,
    DATASET_DIR,
    PLAYERS_MODEL_PATH,
    FIELD_MODEL_PATH,
    DEFAULT_VIDEO_PATH,
    CLEANED_DATASET_DIR,
    MINIMAP_SCALE,
    MINIMAP_PADDING,
    TEAM_COLORS_BGR,
    UNASSIGNED_COLOR,
    CONFIDENCE_THRESHOLD,
    REASSIGN_INTERVAL,
    MAX_COLOR_SAMPLES,
    CLASS_CONFIG,
    CLASS_ID_LOOKUP,
)

# State management
from .state import CombinedState

# Model utilities
from .models import (
    load_players_model,
    load_field_model,
    get_model_label,
)

# Player detection and clustering
from .players import (
    extract_jersey_color,
    collect_initial_team_colors,
    assign_player_team,
    reassign_all_teams,
    resolve_class_metadata,
)

# Field detection and homography
from .field import (
    KEYPOINT_NAMES,
    SoccerPitchConfiguration,
    ViewTransformer,
    detect_field_keypoints,
    compute_view_transformers,
    validate_homography,
    should_update_homography,
    check_keypoint_distribution,
    HOMOGRAPHY_INTERVAL,
    MIN_KEYPOINTS_STABLE,
    MAX_REPROJ_ERROR,
)

# Visualization
from .visualization import (
    bgr_to_rgb_norm,
    draw_pitch,
    compose_split_view,
    minimap_coords,
    create_annotators,
)

# Main pipeline
from .pipeline import (
    initialize_combined_state,
    create_state_from_frame,
    process_combined_frame,
)

# Public API
__all__ = [
    # Version
    "__version__",
    # Config
    "PROJECT_ROOT",
    "MODELS_DIR",
    "DATA_DIR",
    "DATASET_DIR",
    "PLAYERS_MODEL_PATH",
    "FIELD_MODEL_PATH",
    "DEFAULT_VIDEO_PATH",
    "CLEANED_DATASET_DIR",
    "MINIMAP_SCALE",
    "MINIMAP_PADDING",
    "TEAM_COLORS_BGR",
    "UNASSIGNED_COLOR",
    "CONFIDENCE_THRESHOLD",
    "REASSIGN_INTERVAL",
    "MAX_COLOR_SAMPLES",
    "CLASS_CONFIG",
    "CLASS_ID_LOOKUP",
    # State
    "CombinedState",
    # Models
    "load_players_model",
    "load_field_model",
    "get_model_label",
    # Players
    "extract_jersey_color",
    "collect_initial_team_colors",
    "assign_player_team",
    "reassign_all_teams",
    "resolve_class_metadata",
    # Field
    "KEYPOINT_NAMES",
    "SoccerPitchConfiguration",
    "ViewTransformer",
    "detect_field_keypoints",
    "compute_view_transformers",
    "validate_homography",
    "should_update_homography",
    "check_keypoint_distribution",
    "HOMOGRAPHY_INTERVAL",
    "MIN_KEYPOINTS_STABLE",
    "MAX_REPROJ_ERROR",
    # Visualization
    "bgr_to_rgb_norm",
    "draw_pitch",
    "compose_split_view",
    "minimap_coords",
    "create_annotators",
    # Pipeline
    "initialize_combined_state",
    "create_state_from_frame",
    "process_combined_frame",
]
