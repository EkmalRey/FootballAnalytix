"""
State management classes for the FootballAnalytix pipeline.

Contains dataclasses for tracking processing state across frames,
including team assignments, color caches, and view transformers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .field import ViewTransformer


@dataclass
class CombinedState:
    """
    State container for combined player and field processing.
    
    Tracks team color assignments, player tracking data, and 
    homography transformers across video frames.
    
    Attributes:
        team_color_centers: KMeans cluster centers for team colors (RGB)
        color_samples: List of collected jersey color samples
        player_teams: Mapping of track ID to team index
        player_colors_cache: Cached jersey colors per track ID
        pitch_to_frame: Homography transformer from pitch to frame coordinates
        frame_to_pitch: Homography transformer from frame to pitch coordinates
        last_frame_points: Last detected frame keypoints
        last_pitch_points: Corresponding pitch keypoints
    """
    team_color_centers: Optional[np.ndarray] = None
    color_samples: List[np.ndarray] = field(default_factory=list)
    player_teams: Dict[int, int] = field(default_factory=dict)
    player_colors_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    pitch_to_frame: Optional["ViewTransformer"] = None
    frame_to_pitch: Optional["ViewTransformer"] = None
    last_frame_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    last_pitch_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    def reset_tracking(self) -> None:
        """Reset all tracking state (useful when scene changes significantly)."""
        self.player_teams.clear()
        self.player_colors_cache.clear()
        self.color_samples.clear()
    
    def has_team_colors(self) -> bool:
        """Check if team colors have been initialized."""
        return self.team_color_centers is not None
    
    def has_homography(self) -> bool:
        """Check if homography transformers are available."""
        return self.frame_to_pitch is not None and self.pitch_to_frame is not None
    
    def get_team_count(self, team: int) -> int:
        """Count players assigned to a specific team."""
        return sum(1 for t in self.player_teams.values() if t == team)
