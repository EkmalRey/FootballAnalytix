"""
Worker Threads for FootballAnalytix GUI.

Provides thread-safe workers for:
- Model loading on startup
- Single frame inference with debouncing
- Continuous video inference with proper pause/stop
"""

from PyQt6.QtCore import QThread, pyqtSignal, QObject, QMutex, QMutexLocker, QTimer
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import time
import threading
from queue import Queue, Empty

from .controllers.inference_controller import InferenceController


class InferenceWorker(QObject):
    """
    Worker object for handling single-frame inference in a separate thread.
    
    Features:
    - Cancellation of stale requests
    - Only processes the most recent frame
    """
    finished = pyqtSignal(dict)  # Emits result dictionary
    error = pyqtSignal(str)
    
    def __init__(self, controller: InferenceController):
        super().__init__()
        self.controller = controller
        self._current_request_id = 0
        self._lock = QMutex()
        
    def process_frame(self, frame: np.ndarray, frame_idx: int, file_path: Optional[Path], request_id: int = 0):
        """
        Run inference on a frame.
        
        Args:
            frame: The frame to process
            frame_idx: Frame index
            file_path: Path to video file
            request_id: Unique request ID - if a newer request comes in, skip this one
        """
        # Check if this request is stale
        with QMutexLocker(self._lock):
            if request_id < self._current_request_id:
                # A newer request has come in, skip this one
                return
        
        try:
            # Lazy init
            if self.controller.state is None and file_path:
                self.controller.initialize_from_frame(frame, file_path)
            
            # Check again before heavy processing
            with QMutexLocker(self._lock):
                if request_id < self._current_request_id:
                    return
                
            result = self.controller.process_frame(frame, frame_idx)
            
            # Check one more time before emitting
            with QMutexLocker(self._lock):
                if request_id < self._current_request_id:
                    return
            
            if result:
                self.finished.emit(result)
            else:
                self.error.emit("Processing returned no result")
                
        except Exception as e:
            self.error.emit(str(e))
    
    def set_request_id(self, request_id: int):
        """Set the current request ID to cancel older requests."""
        with QMutexLocker(self._lock):
            self._current_request_id = request_id


class ContinuousInferenceWorker(QThread):
    """
    Worker thread for continuous video inference.
    
    Features:
    - Immediate pause/stop response
    - Progress tracking
    - Frame caching
    """
    
    # Signals
    frame_processed = pyqtSignal(int, dict)   # frame_idx, result
    progress = pyqtSignal(int, int, float)    # current, total, fps
    status_changed = pyqtSignal(str)          # status message
    error = pyqtSignal(str)
    finished_all = pyqtSignal()               # Completed all frames
    paused = pyqtSignal()                     # Emitted when actually paused
    
    def __init__(self, controller: InferenceController):
        super().__init__()
        self.controller = controller
        
        # State
        self._video_path: Optional[Path] = None
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._total_frames = 0
        self._fps = 30.0
        self._current_frame_idx = 0
        self._start_frame_idx = 0
        
        # Control flags - use threading.Event for fast response
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._is_paused = threading.Event()  # Signals when actually paused
        
        # Performance
        self._frames_processed = 0
        self._start_time = 0.0
        
    def set_video(self, video_path: Path):
        """Set the video to process."""
        self._video_path = video_path
        
    def set_start_frame(self, frame_idx: int):
        """Set the starting frame index."""
        self._start_frame_idx = frame_idx
        
    def request_stop(self):
        """Request immediate stop."""
        self._stop_flag.set()
        self._pause_flag.set()  # Also unblock pause wait
        
    def request_pause(self):
        """Request immediate pause."""
        self._pause_flag.set()
        
    def request_resume(self):
        """Resume processing."""
        self._is_paused.clear()
        self._pause_flag.clear()
        
    def is_running(self) -> bool:
        """Check if worker thread is alive."""
        return self.isRunning()
    
    def is_paused(self) -> bool:
        """Check if worker is currently paused."""
        return self._is_paused.is_set()
    
    def wait_for_pause(self, timeout_ms: int = 1000) -> bool:
        """Wait until worker is actually paused."""
        return self._is_paused.wait(timeout=timeout_ms / 1000.0)
        
    def run(self):
        """Main processing loop with immediate stop/pause response."""
        if self._video_path is None:
            self.error.emit("No video path set")
            return
        
        # Reset flags
        self._stop_flag.clear()
        self._pause_flag.clear()
        self._is_paused.clear()
        self._frames_processed = 0
        self._start_time = time.time()
        
        try:
            # Open video
            self._video_capture = cv2.VideoCapture(str(self._video_path))
            if not self._video_capture.isOpened():
                self.error.emit(f"Cannot open video: {self._video_path}")
                return
                
            self._total_frames = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = self._video_capture.get(cv2.CAP_PROP_FPS) or 30.0
            
            # Initialize state from video if not done
            if self.controller.state is None:
                self.status_changed.emit("Initializing...")
                self.controller.initialize_from_video(self._video_path)
            
            # Seek to start frame
            self._current_frame_idx = self._start_frame_idx
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame_idx)
            
            self.status_changed.emit("Processing...")
            
            last_fps_time = time.time()
            last_fps_count = 0
            current_fps = 0.0
            
            while self._current_frame_idx < self._total_frames:
                # CHECK STOP FIRST - immediate response
                if self._stop_flag.is_set():
                    self.status_changed.emit("Stopped")
                    break
                
                # CHECK PAUSE - immediate response
                if self._pause_flag.is_set():
                    self._is_paused.set()
                    self.status_changed.emit("Paused")
                    self.paused.emit()
                    
                    # Wait for resume or stop
                    while self._pause_flag.is_set():
                        if self._stop_flag.is_set():
                            break
                        time.sleep(0.05)  # Small sleep to avoid busy-wait
                    
                    if self._stop_flag.is_set():
                        break
                        
                    self._is_paused.clear()
                    self.status_changed.emit("Processing...")
                
                # Read frame
                ret, frame = self._video_capture.read()
                if not ret:
                    self._current_frame_idx += 1
                    continue
                
                # Process frame - check stop flag before heavy work
                if self._stop_flag.is_set():
                    break
                    
                try:
                    result = self.controller.process_frame(frame, self._current_frame_idx)
                    
                    # Check stop again after processing
                    if self._stop_flag.is_set():
                        break
                        
                    if result:
                        self.frame_processed.emit(self._current_frame_idx, result)
                except Exception as e:
                    self.error.emit(f"Frame {self._current_frame_idx}: {e}")
                
                # Update counters
                self._frames_processed += 1
                self._current_frame_idx += 1
                
                # Calculate FPS every second
                now = time.time()
                if now - last_fps_time >= 1.0:
                    current_fps = (self._frames_processed - last_fps_count) / (now - last_fps_time)
                    last_fps_time = now
                    last_fps_count = self._frames_processed
                
                # Emit progress
                self.progress.emit(self._current_frame_idx, self._total_frames, current_fps)
            
            # Completed successfully
            if not self._stop_flag.is_set() and self._current_frame_idx >= self._total_frames:
                self.status_changed.emit("Completed")
                self.finished_all.emit()
                
        except Exception as e:
            self.error.emit(f"Worker error: {e}")
        finally:
            if self._video_capture is not None:
                self._video_capture.release()
                self._video_capture = None


class ModelLoaderWorker(QThread):
    """
    Worker thread specifically for loading models on startup.
    """
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(str)        # status message
    
    def __init__(self, controller: InferenceController):
        super().__init__()
        self.controller = controller
        
    def run(self):
        try:
            self.progress.emit("Loading player detection model...")
            success = self.controller.load_models()
            
            if success:
                self.finished.emit(True, "Models loaded successfully")
            else:
                self.finished.emit(False, "Failed to load models")
        except Exception as e:
            self.finished.emit(False, str(e))
