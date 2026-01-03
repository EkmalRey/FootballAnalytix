"""
Model loading utilities for FootballAnalytix pipeline.

Provides functions to load YOLO models for player detection 
and field keypoint detection with proper error handling.
"""

from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO

from .config import (
    PLAYERS_MODEL_PATH,
    FIELD_MODEL_PATH,
)


def _get_yolo_class():
    """Lazy import of YOLO to avoid import errors when ultralytics is not installed."""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for model loading. "
            "Install with: pip install ultralytics"
        )




def load_players_model(
    model_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> "YOLO":
    """
    Load the YOLO model for player/ball/referee detection.
    
    Args:
        model_path: Path to model file. Uses default if None.
        verbose: Print loading message if True.
    
    Returns:
        Loaded YOLO model instance.
    
    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    path = Path(model_path) if model_path else PLAYERS_MODEL_PATH
    
    if not path.exists():
        raise FileNotFoundError(
            f"Players model not found at {path}. "
            f"Please ensure the model file exists."
        )
    
    YOLO = _get_yolo_class()
    model = YOLO(str(path))
    if verbose:
        print(f"Loaded players model from {path}")
    
    return model


def load_field_model(
    model_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> "YOLO":
    """
    Load the YOLO model for field keypoint detection.
    
    Args:
        model_path: Path to model file. Uses default if None.
        verbose: Print loading message if True.
    
    Returns:
        Loaded YOLO model instance.
    
    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    path = Path(model_path) if model_path else FIELD_MODEL_PATH
    
    if not path.exists():
        raise FileNotFoundError(
            f"Field model not found at {path}. "
            f"Please ensure the model file exists."
        )
    
    YOLO = _get_yolo_class()
    model = YOLO(str(path))
    if verbose:
        print(f"Loaded field model from {path}")
    
    return model


def get_model_label(model: Any, class_id: int) -> str:
    """
    Return a readable class label from a YOLO model, with safe fallbacks.
    
    Args:
        model: YOLO model instance
        class_id: Class ID to look up
    
    Returns:
        Human-readable class label string.
    """
    names = getattr(getattr(model, "model", None), "names", None)
    if names is None:
        names = getattr(model, "names", None)
    
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    
    return f"class_{class_id}"
