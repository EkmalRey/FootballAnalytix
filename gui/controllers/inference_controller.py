"""
Inference Controller for FootballAnalytix GUI.

Wraps the footballanalytix library functions for use in the GUI,
handling model loading, state management, and frame processing.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import threading
import warnings
import logging
import os
import cv2
import numpy as np

# Suppress ultralytics warnings before importing
os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*inference results will accumulate.*')

# Import from the footballanalytix library
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from footballanalytix import (
    # Config
    MODELS_DIR,
    CONFIDENCE_THRESHOLD,
    
    # Models
    load_players_model,
    load_field_model,
    
    # State
    CombinedState,
    
    # Field
    SoccerPitchConfiguration,
    
    # Pipeline
    create_state_from_frame,
    initialize_combined_state,
    process_combined_frame,
    
    # Visualization
    draw_pitch,
)


@dataclass
class ProcessingConfig:
    """Configuration for frame processing."""
    confidence: float = 0.65
    show_object_detection: bool = True
    show_clustering: bool = True
    show_id: bool = True
    show_detected_keypoints: bool = True
    
    # Pitch configuration
    pitch_length: int = 12000
    pitch_width: int = 7000
    penalty_box_length: int = 1886
    penalty_box_width: int = 4140
    goal_box_length: int = 629
    goal_box_width: int = 1885
    center_circle_radius: int = 942
    penalty_spot_distance: int = 1257


class InferenceController:
    """
    Controller for managing inference operations.
    
    Handles model loading, state management, and frame processing
    in a thread-safe manner.
    """
    
    def __init__(self, on_status_update: Optional[Callable[[str], None]] = None):
        """
        Initialize the inference controller.
        
        Args:
            on_status_update: Optional callback for status updates
        """
        self.on_status_update = on_status_update or (lambda x: None)
        
        # Models
        self.players_model = None
        self.field_model = None
        self.models_loaded = False
        
        # State
        self.state: Optional[CombinedState] = None
        self.pitch_config: Optional[SoccerPitchConfiguration] = None
        self.base_minimap: Optional[np.ndarray] = None
        
        # Configuration
        self.config = ProcessingConfig()
        
        # Threading
        self._loading_lock = threading.Lock()
        self._processing_lock = threading.Lock()
    
    def _update_status(self, message: str):
        """Send status update via callback."""
        self.on_status_update(message)
    
    def load_models(self) -> bool:
        """
        Load player and field detection models.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        with self._loading_lock:
            try:
                self._update_status("Loading player detection model...")
                self.players_model = load_players_model()
                
                self._update_status("Loading field detection model...")
                self.field_model = load_field_model()
                
                self.models_loaded = True
                self._update_status("Models loaded successfully!")
                return True
                
            except Exception as e:
                self._update_status(f"Error loading models: {e}")
                self.models_loaded = False
                return False
    
    def initialize_pitch(self):
        """Initialize pitch configuration and base minimap."""
        self.pitch_config = SoccerPitchConfiguration(
            length=self.config.pitch_length,
            width=self.config.pitch_width,
            penalty_box_length=self.config.penalty_box_length,
            penalty_box_width=self.config.penalty_box_width,
            goal_box_length=self.config.goal_box_length,
            goal_box_width=self.config.goal_box_width,
            centre_circle_radius=self.config.center_circle_radius,
            penalty_spot_distance=self.config.penalty_spot_distance,
        )
        self.base_minimap = draw_pitch(self.pitch_config)
    
    def initialize_from_video(self, video_path: Path, num_frames: int = 30) -> bool:
        """
        Initialize state from a video file for team color detection.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample for color detection
            
        Returns:
            True if initialization successful
        """
        if not self.models_loaded:
            self._update_status("Models not loaded!")
            return False
        
        try:
            self._update_status("Initializing from video...")
            self.state = initialize_combined_state(
                video_path=video_path,
                model=self.players_model,
                num_frames=num_frames,
                conf=self.config.confidence,
            )
            self.initialize_pitch()
            self._update_status("Video initialized!")
            return True
            
        except Exception as e:
            self._update_status(f"Error initializing video: {e}")
            return False
    
    def initialize_from_frame(self, frame: np.ndarray, image_path: Optional[Path] = None) -> bool:
        """
        Initialize state from a single frame.
        
        Args:
            frame: BGR image
            image_path: Optional path to image file
            
        Returns:
            True if initialization successful
        """
        if not self.models_loaded:
            self._update_status("Models not loaded!")
            return False
        
        try:
            self._update_status("Initializing from frame...")
            self.state = create_state_from_frame(
                frame=frame,
                model=self.players_model,
                conf=self.config.confidence,
                image_path=image_path,
            )
            self.initialize_pitch()
            self._update_status("Frame initialized!")
            return True
            
        except Exception as e:
            self._update_status(f"Error initializing frame: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Optional[Dict[str, Any]]:
        """
        Process a single frame with detection and visualization.
        
        Args:
            frame: BGR image to process
            frame_idx: Frame index for tracking
            
        Returns:
            Dictionary with 'annotated_frame', 'minimap', and 'stats', or None on error
        """
        if not self.models_loaded or self.state is None:
            return None
        
        with self._processing_lock:
            try:
                result = process_combined_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    state=self.state,
                    players_model=self.players_model,
                    field_model=self.field_model,
                    config=self.pitch_config,
                    conf=self.config.confidence,
                    show_object_detection=self.config.show_object_detection,
                    show_clustering=self.config.show_clustering,
                    show_id=self.config.show_id,
                    show_detected_vertices=self.config.show_detected_keypoints,
                    base_minimap=self.base_minimap.copy() if self.base_minimap is not None else None,
                )
                return result
                
            except Exception as e:
                self._update_status(f"Error processing frame: {e}")
                return None
    
    def update_config(self, **kwargs):
        """Update processing configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_config(self) -> ProcessingConfig:
        """Get current processing configuration."""
        return self.config
