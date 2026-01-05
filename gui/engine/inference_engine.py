"""
Core Inference Engine for FootballAnalytix GUI.

Provides a stateful engine that manages:
- Video capture and frame reading
- Inference processing via controller
- Frame caching for replay
- Live configuration updates
- Start/Stop/Pause controls
"""

from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
from enum import Enum, auto
from dataclasses import dataclass
import threading
import time
import hashlib
import json
import cv2
import numpy as np

from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

from .frame_cache import FrameCache, CachedFrame
from ..controllers.inference_controller import InferenceController, ProcessingConfig


class EngineState(Enum):
    """Engine operational states."""
    IDLE = auto()       # No video loaded
    READY = auto()      # Video loaded, not processing
    RUNNING = auto()    # Actively processing frames
    PAUSED = auto()     # Processing paused, can resume
    STOPPED = auto()    # Processing stopped, can restart


@dataclass
class VideoInfo:
    """Information about the loaded video."""
    path: Path
    total_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float


class InferenceEngine(QObject):
    """
    Core inference engine managing video processing pipeline.
    
    Signals:
        frame_ready: Emitted when a frame is processed (frame_idx, result_dict)
        progress_updated: Emitted with processing progress (current_frame, total_frames, fps)
        state_changed: Emitted when engine state changes (new_state)
        error_occurred: Emitted on errors (error_message)
        stats_updated: Emitted with live stats (stats_dict)
    """
    
    # Signals
    frame_ready = pyqtSignal(int, dict)       # frame_idx, result
    progress_updated = pyqtSignal(int, int, float)  # current, total, processing_fps
    state_changed = pyqtSignal(EngineState)
    error_occurred = pyqtSignal(str)
    stats_updated = pyqtSignal(dict)
    
    def __init__(self, controller: InferenceController, cache_size: int = 500):
        """
        Initialize the inference engine.
        
        Args:
            controller: InferenceController for running inference
            cache_size: Maximum frames to cache
        """
        super().__init__()
        
        self.controller = controller
        self.frame_cache = FrameCache(max_size=cache_size)
        
        # Video state
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._video_info: Optional[VideoInfo] = None
        self._current_frame_idx: int = 0
        
        # Engine state
        self._state = EngineState.IDLE
        self._state_lock = QMutex()
        
        # Processing control
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self._frames_processed = 0
        self._processing_start_time = 0.0
        self._last_fps_calc_time = 0.0
        self._last_fps_frame_count = 0
        self._current_processing_fps = 0.0
    
    @property
    def state(self) -> EngineState:
        """Get current engine state."""
        with QMutexLocker(self._state_lock):
            return self._state
    
    @state.setter
    def state(self, new_state: EngineState):
        """Set engine state and emit signal."""
        with QMutexLocker(self._state_lock):
            if self._state != new_state:
                self._state = new_state
                self.state_changed.emit(new_state)
    
    @property
    def video_info(self) -> Optional[VideoInfo]:
        """Get loaded video information."""
        return self._video_info
    
    @property
    def current_frame_idx(self) -> int:
        """Get current frame index."""
        return self._current_frame_idx
    
    def get_config_hash(self) -> str:
        """Generate hash of current processing config for cache validation."""
        config = self.controller.config
        config_dict = {
            'confidence': config.confidence,
            'show_object_detection': config.show_object_detection,
            'show_clustering': config.show_clustering,
            'show_id': config.show_id,
            'show_detected_keypoints': config.show_detected_keypoints,
        }
        return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]
    
    # =========================================================================
    # Video Loading
    # =========================================================================
    
    def load_video(self, video_path: Path) -> bool:
        """
        Load a video file for processing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if loaded successfully
        """
        # Stop any current processing
        self.stop()
        
        try:
            # Release previous capture
            if self._video_capture is not None:
                self._video_capture.release()
            
            # Open new video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.error_occurred.emit(f"Cannot open video: {video_path}")
                return False
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self._video_capture = cap
            self._video_info = VideoInfo(
                path=video_path,
                total_frames=total_frames,
                fps=fps,
                width=width,
                height=height,
                duration_seconds=total_frames / fps if fps > 0 else 0
            )
            
            # Clear cache and reset state
            self.frame_cache.clear()
            self._current_frame_idx = 0
            self._frames_processed = 0
            
            # Initialize state from video
            if self.controller.models_loaded:
                self.controller.initialize_from_video(video_path)
            
            self.state = EngineState.READY
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error loading video: {e}")
            return False
    
    def release_video(self):
        """Release the video capture and clear state."""
        self.stop()
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        self._video_info = None
        self.frame_cache.clear()
        self.state = EngineState.IDLE
    
    # =========================================================================
    # Frame Reading
    # =========================================================================
    
    def read_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read a specific frame from the video.
        
        Args:
            frame_idx: Frame index to read
            
        Returns:
            BGR frame or None if failed
        """
        if self._video_capture is None or self._video_info is None:
            return None
        
        if frame_idx < 0 or frame_idx >= self._video_info.total_frames:
            return None
        
        # Check cache first
        cached = self.frame_cache.get(frame_idx)
        if cached is not None:
            return cached.raw_frame.copy()
        
        # Seek and read
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._video_capture.read()
        
        if ret:
            return frame
        return None
    
    def get_cached_result(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get cached processing result for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Result dict or None if not cached
        """
        cached = self.frame_cache.get(frame_idx)
        if cached is not None and cached.annotated_frame is not None:
            return {
                'frame': cached.annotated_frame,
                'minimap': cached.minimap,
                'stats': cached.stats,
                'from_cache': True
            }
        return None
    
    # =========================================================================
    # Processing Control
    # =========================================================================
    
    def start(self, from_frame: Optional[int] = None) -> bool:
        """
        Start or resume inference processing.
        
        Args:
            from_frame: Start from this frame (default: current position)
            
        Returns:
            True if started successfully
        """
        if self._video_info is None:
            self.error_occurred.emit("No video loaded")
            return False
        
        if not self.controller.models_loaded:
            self.error_occurred.emit("Models not loaded")
            return False
        
        # Set starting frame
        if from_frame is not None:
            self._current_frame_idx = max(0, min(from_frame, self._video_info.total_frames - 1))
        
        # Reset control flags
        self._stop_requested.clear()
        self._pause_requested.clear()
        
        # Start processing thread
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_start_time = time.time()
        self._last_fps_calc_time = time.time()
        self._last_fps_frame_count = 0
        
        self.state = EngineState.RUNNING
        self._processing_thread.start()
        
        return True
    
    def pause(self):
        """Pause processing (can be resumed)."""
        if self.state == EngineState.RUNNING:
            self._pause_requested.set()
    
    def resume(self):
        """Resume paused processing."""
        if self.state == EngineState.PAUSED:
            self._pause_requested.clear()
            self.start(from_frame=self._current_frame_idx)
    
    def stop(self):
        """Stop processing completely."""
        self._stop_requested.set()
        self._pause_requested.set()  # Unblock pause wait
        
        if self._processing_thread is not None and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
        
        if self.state in (EngineState.RUNNING, EngineState.PAUSED):
            self.state = EngineState.STOPPED if self._video_info else EngineState.IDLE
    
    def _processing_loop(self):
        """Main processing loop running in background thread."""
        try:
            while self._current_frame_idx < self._video_info.total_frames:
                # Check stop
                if self._stop_requested.is_set():
                    break
                
                # Check pause
                if self._pause_requested.is_set():
                    self.state = EngineState.PAUSED
                    while self._pause_requested.is_set() and not self._stop_requested.is_set():
                        time.sleep(0.1)
                    if self._stop_requested.is_set():
                        break
                    self.state = EngineState.RUNNING
                
                # Read frame
                frame = self.read_frame_at(self._current_frame_idx)
                if frame is None:
                    self._current_frame_idx += 1
                    continue
                
                # Process frame
                result = self._process_single_frame(frame, self._current_frame_idx)
                
                if result:
                    # Cache the result
                    self.frame_cache.put(
                        self._current_frame_idx,
                        frame,
                        result.get('frame'),
                        result.get('minimap'),
                        result.get('stats'),
                        self.get_config_hash()
                    )
                    
                    # Emit signals
                    self.frame_ready.emit(self._current_frame_idx, result)
                    
                    if result.get('stats'):
                        self.stats_updated.emit(result['stats'])
                
                # Update progress and FPS
                self._frames_processed += 1
                self._update_processing_fps()
                self.progress_updated.emit(
                    self._current_frame_idx,
                    self._video_info.total_frames,
                    self._current_processing_fps
                )
                
                self._current_frame_idx += 1
            
            # Completed
            if not self._stop_requested.is_set():
                self.state = EngineState.READY
                
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {e}")
            self.state = EngineState.STOPPED
    
    def _process_single_frame(self, frame: np.ndarray, frame_idx: int) -> Optional[Dict[str, Any]]:
        """Process a single frame through the controller."""
        try:
            return self.controller.process_frame(frame, frame_idx)
        except Exception as e:
            self.error_occurred.emit(f"Frame {frame_idx} error: {e}")
            return None
    
    def _update_processing_fps(self):
        """Calculate and update processing FPS."""
        current_time = time.time()
        elapsed = current_time - self._last_fps_calc_time
        
        if elapsed >= 1.0:  # Update every second
            frames_since_last = self._frames_processed - self._last_fps_frame_count
            self._current_processing_fps = frames_since_last / elapsed
            self._last_fps_calc_time = current_time
            self._last_fps_frame_count = self._frames_processed
    
    # =========================================================================
    # Single Frame Processing (for manual navigation)
    # =========================================================================
    
    def process_frame(self, frame_idx: int, force_reprocess: bool = False) -> Optional[Dict[str, Any]]:
        """
        Process a single frame (for manual navigation or live preview).
        
        Args:
            frame_idx: Frame index to process
            force_reprocess: If True, ignore cache and reprocess
            
        Returns:
            Processing result dict or None
        """
        if self._video_info is None:
            return None
        
        # Check cache (with config validation)
        if not force_reprocess:
            cached = self.frame_cache.get(frame_idx)
            if cached is not None and cached.annotated_frame is not None:
                # Check if config matches
                if cached.config_hash == self.get_config_hash():
                    return {
                        'frame': cached.annotated_frame,
                        'minimap': cached.minimap,
                        'stats': cached.stats,
                        'from_cache': True
                    }
        
        # Read and process frame
        frame = self.read_frame_at(frame_idx)
        if frame is None:
            return None
        
        result = self._process_single_frame(frame, frame_idx)
        
        if result:
            # Cache the result
            self.frame_cache.put(
                frame_idx,
                frame,
                result.get('frame'),
                result.get('minimap'),
                result.get('stats'),
                self.get_config_hash()
            )
            
            if result.get('stats'):
                self.stats_updated.emit(result['stats'])
        
        return result
    
    # =========================================================================
    # Re-rendering (for live config changes)
    # =========================================================================
    
    def rerender_frame(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        """
        Re-render a cached frame with current visualization settings.
        
        This re-runs inference to apply new settings like toggling
        object detection visibility, team clustering, etc.
        
        Args:
            frame_idx: Frame index to re-render
            
        Returns:
            New result dict or None
        """
        return self.process_frame(frame_idx, force_reprocess=True)
    
    # =========================================================================
    # Navigation
    # =========================================================================
    
    def seek(self, frame_idx: int) -> bool:
        """
        Seek to a specific frame.
        
        Args:
            frame_idx: Target frame index
            
        Returns:
            True if successful
        """
        if self._video_info is None:
            return False
        
        frame_idx = max(0, min(frame_idx, self._video_info.total_frames - 1))
        self._current_frame_idx = frame_idx
        return True
    
    def next_frame(self) -> Optional[int]:
        """Move to next frame. Returns new frame index or None."""
        if self._video_info is None:
            return None
        if self._current_frame_idx < self._video_info.total_frames - 1:
            self._current_frame_idx += 1
            return self._current_frame_idx
        return None
    
    def prev_frame(self) -> Optional[int]:
        """Move to previous frame. Returns new frame index or None."""
        if self._video_info is None:
            return None
        if self._current_frame_idx > 0:
            self._current_frame_idx -= 1
            return self._current_frame_idx
        return None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            'frames_processed': self._frames_processed,
            'cached_frames': len(self.frame_cache),
            'processing_fps': self._current_processing_fps,
            'total_frames': self._video_info.total_frames if self._video_info else 0,
            'progress_percent': (
                (self._frames_processed / self._video_info.total_frames * 100)
                if self._video_info and self._video_info.total_frames > 0 else 0
            )
        }
