"""
Video Player Component for displaying images and video frames.
"""

from pathlib import Path
from typing import Optional, Callable, Tuple
import threading
import time
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image

from ..utils.image_utils import cv2_to_pil, resize_to_fit


class VideoPlayer(ctk.CTkFrame):
    """
    Video player component for displaying processed frames.
    """
    
    def __init__(
        self,
        master,
        width: int = 800,
        height: int = 500,
        on_frame_changed: Optional[Callable[[int], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        
        self.display_width = width
        self.display_height = height
        self.on_frame_changed = on_frame_changed
        
        # Video state
        self.video_path: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.current_frame_idx: int = 0
        self.fps: float = 30.0
        
        # Playback state
        self.is_playing: bool = False
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_playback = threading.Event()
        
        # Current frame data
        self.current_frame: Optional[np.ndarray] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the video player UI."""
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Canvas for video display
        self.canvas = ctk.CTkLabel(
            self,
            text="No media loaded",
            width=self.display_width,
            height=self.display_height,
            fg_color=("gray85", "gray20"),
            corner_radius=8
        )
        self.canvas.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Controls frame
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.controls_frame.grid_columnconfigure(3, weight=1)
        
        # Playback controls
        self.prev_btn = ctk.CTkButton(
            self.controls_frame,
            text="⏮",
            width=40,
            command=self._prev_frame
        )
        self.prev_btn.grid(row=0, column=0, padx=2)
        
        self.play_btn = ctk.CTkButton(
            self.controls_frame,
            text="▶",
            width=40,
            command=self._toggle_play
        )
        self.play_btn.grid(row=0, column=1, padx=2)
        
        self.next_btn = ctk.CTkButton(
            self.controls_frame,
            text="⏭",
            width=40,
            command=self._next_frame
        )
        self.next_btn.grid(row=0, column=2, padx=2)
        
        # Frame counter
        self.frame_label = ctk.CTkLabel(
            self.controls_frame,
            text="Frame: 0/0"
        )
        self.frame_label.grid(row=0, column=3, padx=10, sticky="w")
        
        # Frame slider
        self.frame_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=0,
            to=100,
            command=self._on_slider_change
        )
        self.frame_slider.grid(row=0, column=4, padx=10, sticky="ew")
        self.controls_frame.grid_columnconfigure(4, weight=1)
    
    def load_image(self, path: Path) -> bool:
        """Load a single image."""
        try:
            self.video_path = path
            self.cap = None
            self.total_frames = 1
            self.current_frame_idx = 0
            
            # Read image
            self.current_frame = cv2.imread(str(path))
            if self.current_frame is None:
                return False
            
            # Update display
            self._display_frame(self.current_frame)
            self._update_controls()
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def load_video(self, path: Path) -> bool:
        """Load a video file."""
        try:
            self.video_path = path
            self.cap = cv2.VideoCapture(str(path))
            
            if not self.cap.isOpened():
                return False
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.current_frame_idx = 0
            
            # Read first frame
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self._display_frame(frame)
            
            self._update_controls()
            return True
            
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
    
    def get_frame(self, idx: int) -> Optional[np.ndarray]:
        """Get a specific frame by index."""
        if self.cap is None:
            return self.current_frame
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.current_frame_idx = idx
            return frame
        return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame."""
        return self.current_frame
    
    def display_processed_frame(self, frame: np.ndarray):
        """Display a processed frame (after inference)."""
        self._display_frame(frame)
    
    def _display_frame(self, frame: np.ndarray):
        """Display a frame on the canvas."""
        try:
            # Resize to fit
            resized, (w, h) = resize_to_fit(frame, self.display_width, self.display_height)
            
            # Convert to PIL and CTkImage
            pil_image = cv2_to_pil(resized)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(w, h))
            
            # Update canvas
            self.canvas.configure(image=ctk_image, text="")
            self.canvas.image = ctk_image  # Keep reference
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def _update_controls(self):
        """Update control states."""
        self.frame_label.configure(text=f"Frame: {self.current_frame_idx + 1}/{self.total_frames}")
        
        if self.total_frames > 1:
            self.frame_slider.configure(to=self.total_frames - 1)
            self.frame_slider.set(self.current_frame_idx)
        else:
            self.frame_slider.configure(to=1)
            self.frame_slider.set(0)
    
    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            self._seek_frame(self.current_frame_idx - 1)
    
    def _next_frame(self):
        """Go to next frame."""
        if self.current_frame_idx < self.total_frames - 1:
            self._seek_frame(self.current_frame_idx + 1)
    
    def _seek_frame(self, idx: int):
        """Seek to a specific frame."""
        frame = self.get_frame(idx)
        if frame is not None:
            self._display_frame(frame)  # Always display raw frame immediately
            self._update_controls()
            if self.on_frame_changed:
                self.on_frame_changed(idx)
    
    def _on_slider_change(self, value):
        """Handle slider change."""
        idx = int(value)
        if idx != self.current_frame_idx:
            self._seek_frame(idx)
    
    def _toggle_play(self):
        """Toggle playback."""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """Start playback."""
        if self.cap is None or self.total_frames <= 1:
            return
        
        self.is_playing = True
        self.play_btn.configure(text="⏸")
        self._play_next_frame()
    
    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self.play_btn.configure(text="▶")
        if hasattr(self, '_playback_start_time'):
            del self._playback_start_time
    
    def _play_next_frame(self):
        """Play next frame loop with frame dropping for sync."""
        if not self.is_playing:
            return
            
        start_time = time.time()
        
        # Calculate expected frame based on time elapsed
        if not hasattr(self, '_playback_start_time'):
            self._playback_start_time = start_time
            self._start_frame_idx = self.current_frame_idx
        
        elapsed_since_start = start_time - self._playback_start_time
        target_frame_offset = int(elapsed_since_start * self.fps)
        target_frame = self._start_frame_idx + target_frame_offset
        
        if target_frame >= self.total_frames:
            self.pause()
            if hasattr(self, '_playback_start_time'):
                del self._playback_start_time
            return
            
        # If we are behind, skip frames. If ahead, wait.
        if target_frame > self.current_frame_idx:
            # We are behind or on time
            self._seek_frame(target_frame)
            
            # Schedule next check quickly to keep up
            # overhead compensation
            overhead = (time.time() - start_time) * 1000
            delay = max(1, int((1000/self.fps) - overhead))
            self.after(delay, self._play_next_frame)
        else:
            # We are ahead (shouldn't happen often with simple logic but safe to handle)
            self._seek_frame(self.current_frame_idx + 1)
            self.after(int(1000/self.fps), self._play_next_frame)
    
    def release(self):
        """Release video resources."""
        self.pause()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
