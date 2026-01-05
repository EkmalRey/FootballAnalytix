"""
Frame Cache for storing processed inference results.

Enables replay functionality by storing raw frames and
their processed outputs (annotated frame, minimap, stats).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
from collections import OrderedDict
import numpy as np
import threading


@dataclass
class CachedFrame:
    """Stores all data for a single processed frame."""
    frame_idx: int
    raw_frame: np.ndarray
    annotated_frame: Optional[np.ndarray] = None
    minimap: Optional[np.ndarray] = None
    stats: Optional[Dict[str, Any]] = None
    config_hash: Optional[str] = None  # Hash of config used for this result


class FrameCache:
    """
    LRU Cache for processed video frames.
    
    Stores both raw and processed frames to enable:
    - Replay of previously processed frames
    - Re-rendering with different visualization settings
    - Navigation (prev/next) through processed results
    
    Thread-safe for concurrent access.
    """
    
    def __init__(self, max_size: int = 500):
        """
        Initialize the frame cache.
        
        Args:
            max_size: Maximum number of frames to cache
        """
        self.max_size = max_size
        self._cache: OrderedDict[int, CachedFrame] = OrderedDict()
        self._lock = threading.RLock()
        
        # Track the range of cached frames
        self._min_frame_idx: Optional[int] = None
        self._max_frame_idx: Optional[int] = None
        
    def put(
        self, 
        frame_idx: int, 
        raw_frame: np.ndarray,
        annotated_frame: Optional[np.ndarray] = None,
        minimap: Optional[np.ndarray] = None,
        stats: Optional[Dict[str, Any]] = None,
        config_hash: Optional[str] = None
    ) -> None:
        """
        Store a frame and its processed results.
        
        Args:
            frame_idx: Frame index in the video
            raw_frame: Original unprocessed frame
            annotated_frame: Processed frame with annotations
            minimap: Minimap visualization
            stats: Statistics dictionary
            config_hash: Hash of the config used for processing
        """
        with self._lock:
            # If already exists, move to end (most recently used)
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
                cached = self._cache[frame_idx]
                # Update with new data if provided
                if annotated_frame is not None:
                    cached.annotated_frame = annotated_frame
                if minimap is not None:
                    cached.minimap = minimap
                if stats is not None:
                    cached.stats = stats
                if config_hash is not None:
                    cached.config_hash = config_hash
            else:
                # Evict oldest if at capacity
                while len(self._cache) >= self.max_size:
                    evicted_idx, _ = self._cache.popitem(last=False)
                    # Update min range
                    if self._cache:
                        self._min_frame_idx = next(iter(self._cache))
                    else:
                        self._min_frame_idx = None
                
                # Add new frame
                self._cache[frame_idx] = CachedFrame(
                    frame_idx=frame_idx,
                    raw_frame=raw_frame.copy(),
                    annotated_frame=annotated_frame.copy() if annotated_frame is not None else None,
                    minimap=minimap.copy() if minimap is not None else None,
                    stats=stats.copy() if stats else None,
                    config_hash=config_hash
                )
            
            # Update range tracking
            if self._min_frame_idx is None or frame_idx < self._min_frame_idx:
                self._min_frame_idx = frame_idx
            if self._max_frame_idx is None or frame_idx > self._max_frame_idx:
                self._max_frame_idx = frame_idx
                
    def get(self, frame_idx: int) -> Optional[CachedFrame]:
        """
        Retrieve a cached frame by index.
        
        Args:
            frame_idx: Frame index to retrieve
            
        Returns:
            CachedFrame if found, None otherwise
        """
        with self._lock:
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
                return self._cache[frame_idx]
            return None
    
    def has_processed(self, frame_idx: int, config_hash: Optional[str] = None) -> bool:
        """
        Check if a frame has been processed (and optionally with specific config).
        
        Args:
            frame_idx: Frame index to check
            config_hash: If provided, only return True if config matches
            
        Returns:
            True if frame is cached and processed
        """
        with self._lock:
            cached = self._cache.get(frame_idx)
            if cached is None:
                return False
            if cached.annotated_frame is None:
                return False
            if config_hash is not None and cached.config_hash != config_hash:
                return False
            return True
    
    def get_cached_indices(self) -> List[int]:
        """Get list of all cached frame indices."""
        with self._lock:
            return list(self._cache.keys())
    
    def get_processed_indices(self) -> List[int]:
        """Get list of frame indices that have been fully processed."""
        with self._lock:
            return [
                idx for idx, cached in self._cache.items()
                if cached.annotated_frame is not None
            ]
    
    def get_range(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get the range of cached frames.
        
        Returns:
            Tuple of (min_frame_idx, max_frame_idx), or (None, None) if empty
        """
        with self._lock:
            return (self._min_frame_idx, self._max_frame_idx)
    
    def get_next_unprocessed(self, after_idx: int) -> Optional[int]:
        """
        Find the next unprocessed frame after a given index.
        
        Useful for resuming processing.
        
        Args:
            after_idx: Search for frames after this index
            
        Returns:
            Frame index if found, None otherwise
        """
        with self._lock:
            for idx in sorted(self._cache.keys()):
                if idx > after_idx:
                    cached = self._cache[idx]
                    if cached.annotated_frame is None:
                        return idx
            return None
    
    def clear(self) -> None:
        """Clear all cached frames."""
        with self._lock:
            self._cache.clear()
            self._min_frame_idx = None
            self._max_frame_idx = None
    
    def __len__(self) -> int:
        """Return number of cached frames."""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, frame_idx: int) -> bool:
        """Check if frame index is in cache."""
        with self._lock:
            return frame_idx in self._cache
