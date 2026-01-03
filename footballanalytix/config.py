"""
Configuration and constants for FootballAnalytix pipeline.

This module centralizes all configuration, paths, and constants used 
throughout the pipeline. Paths are relative to the project root for 
portability across different machines/clones.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random

# ---------------------------------------------------------------------------
# Path Configuration (Portable - relative to project root)
# ---------------------------------------------------------------------------

# Project root is parent of this package directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Resource directories
RESOURCES_DIR = PROJECT_ROOT / "Models"
MODELS_DIR = RESOURCES_DIR
DATA_DIR = PROJECT_ROOT / "Data"
DATASET_DIR = DATA_DIR / "Sample"

# Model paths
PLAYERS_MODEL_PATH = MODELS_DIR / "Players.pt"
FIELD_MODEL_PATH = MODELS_DIR / "Fields.pt"

# Default video/data inputs
DEFAULT_VIDEO_PATH = DATASET_DIR / "Video.mp4"
DEFAULT_PITCH_IMAGE = DATASET_DIR / "Pitch.png"
CLEANED_DATASET_DIR = DATASET_DIR / "SNGS-131" / "Frames"

# ---------------------------------------------------------------------------
# Visualization Constants
# ---------------------------------------------------------------------------

MINIMAP_SCALE: float = 0.1
MINIMAP_PADDING: int = 50

# Team colors (BGR format for OpenCV)
TEAM_COLORS_BGR: List[Tuple[int, int, int]] = [
    (0, 0, 255),    # Team 0 - red
    (255, 0, 0),    # Team 1 - blue
]

# Other visualization colors (BGR)
HUD_COLOR: Tuple[int, int, int] = (0, 255, 0)
UNASSIGNED_COLOR: Tuple[int, int, int] = (0, 215, 255)
DEFAULT_OTHER_COLOR: Tuple[int, int, int] = (0, 255, 0)
MINIMAP_KEYPOINT_COLOR: Tuple[int, int, int] = (255, 255, 255)

# ---------------------------------------------------------------------------
# Detection Constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.65
REASSIGN_INTERVAL: int = 300  # frames between team reassignments
MAX_COLOR_SAMPLES: int = 200
DEFAULT_MINIMAP_RADIUS: int = 10

# ---------------------------------------------------------------------------
# Class Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClassMetadata:
    """Metadata for a detection class."""
    ids: List[int]
    color: Tuple[int, int, int]
    legend: str
    is_player: bool = False
    show_on_minimap: bool = True
    minimap_radius: int = DEFAULT_MINIMAP_RADIUS


# Per-class configuration (adjust ids/colors per dataset)
CLASS_CONFIG: Dict[str, Dict[str, Any]] = {
    "player": {
        "ids": [1],
        "color": (0, 0, 255),
        "legend": "Player",
        "is_player": True,
        "show_on_minimap": True,
        "minimap_radius": 12,
    },
    "ball": {
        "ids": [3],
        "color": (0, 255, 255),
        "legend": "Ball",
        "is_player": False,
        "show_on_minimap": True,
        "minimap_radius": 10,
    },
    "referee": {
        "ids": [0],
        "color": (204, 0, 255),
        "legend": "Referee",
        "is_player": False,
        "show_on_minimap": True,
        "minimap_radius": 10,
    },
    "goalkeeper": {
        "ids": [2],
        "color": (0, 165, 255),
        "legend": "Goalkeeper",
        "is_player": False,
        "show_on_minimap": True,
        "minimap_radius": 10,
    }
}

# Build class ID lookup table
CLASS_ID_LOOKUP: Dict[int, str] = {}
for class_name, cfg in CLASS_CONFIG.items():
    cfg.setdefault("legend", class_name.title())
    cfg.setdefault("color", DEFAULT_OTHER_COLOR)
    cfg.setdefault("is_player", False)
    cfg.setdefault("show_on_minimap", True)
    cfg.setdefault("minimap_radius", DEFAULT_MINIMAP_RADIUS)
    for class_id in cfg.get("ids", []):
        CLASS_ID_LOOKUP[int(class_id)] = class_name


def get_class_config(class_name: str) -> Dict[str, Any]:
    """Get configuration for a class by name."""
    return CLASS_CONFIG.get(class_name, {})


def get_class_by_id(class_id: int) -> str:
    """Get class name from class ID."""
    return CLASS_ID_LOOKUP.get(class_id, f"class_{class_id}")


# ---------------------------------------------------------------------------
# Training Configurations
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    data_root: Path = field(default_factory=lambda: PROJECT_ROOT / "Data" / "Sample")
    yolo_root: Path = field(default_factory=lambda: PROJECT_ROOT / "Data" / "yolo_dataset")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "Models" / "Training" / "result")

    use_limited_dataset: bool = True
    max_images_limit: int = 10_000

    debug_train_only: bool = False
    debug_val_count: int = 24
    debug_test_count: int = 12

    fast_training: bool = True
    random_seed: int = 42

    img_size: int = 640
    batch_size: int = 16

    def __post_init__(self) -> None:
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)
        if isinstance(self.yolo_root, str):
            self.yolo_root = Path(self.yolo_root)
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        random.seed(self.random_seed)

    def get_training_params(self) -> Dict[str, Any]:
        if self.fast_training:
            return {"epochs": 50, "model_size": "n", "batch_size": self.batch_size}
        return {"epochs": 100, "model_size": "m", "batch_size": self.batch_size}


@dataclass
class ObjectDetectionConfig(TrainingConfig):
    class_map: Dict[str, int] = field(init=False)
    class_colors: Dict[str, Tuple[int, int, int]] = field(init=False)
    project_name: str = "SISIKRIPSI_objects"

    def __post_init__(self) -> None:
        super().__post_init__()
        # Use centralized CLASS_CONFIG
        self.class_map = {name: cfg['ids'][0] for name, cfg in CLASS_CONFIG.items()}
        self.class_colors = {name: cfg['color'] for name, cfg in CLASS_CONFIG.items()}

    def get_class_names(self) -> List[str]:
        return [k for k, _ in sorted(self.class_map.items(), key=lambda kv: kv[1])]

    def get_yolo_model_name(self) -> str:
        size = self.get_training_params()["model_size"]
        return f"yolo11{size}.pt"


@dataclass
class PoseEstimationConfig(TrainingConfig):
    pitch_keypoints: List[str] = field(default_factory=lambda: [f"kp_{i:02d}" for i in range(1, 33)])
    total_keypoints: int = 32
    canonical_kp_order: List[Tuple[str, int]] = field(default_factory=list)
    project_name: str = "SISIKRIPSI_pitch"
    line_color: Tuple[int, int, int] = (0, 191, 255)

    def get_yolo_model_name(self) -> str:
        size = self.get_training_params()["model_size"]
        return f"yolo11{size}-pose.pt"
