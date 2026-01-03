"""
Visualization utilities for FootballAnalytix pipeline.

Provides functions for drawing the soccer pitch minimap,
composing split views, and color conversion utilities.
"""

from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import supervision as sv

from .config import MINIMAP_SCALE, MINIMAP_PADDING
from .field import SoccerPitchConfiguration


# ---------------------------------------------------------------------------
# Color Utilities
# ---------------------------------------------------------------------------

def bgr_to_rgb_norm(color: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert a BGR color tuple in 0-255 space to normalized RGB for Matplotlib.
    
    Args:
        color: BGR color tuple (0-255)
    
    Returns:
        RGB color tuple (0.0-1.0)
    """
    b, g, r = color
    return (r / 255.0, g / 255.0, b / 255.0)


# ---------------------------------------------------------------------------
# Pitch Drawing
# ---------------------------------------------------------------------------

def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = MINIMAP_PADDING,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = MINIMAP_SCALE,
) -> np.ndarray:
    """
    Render a scaled soccer pitch image.
    
    Args:
        config: Soccer pitch configuration
        background_color: Pitch background color (default: green)
        line_color: Line color (default: white)
        padding: Padding around the pitch
        line_thickness: Thickness of pitch lines
        point_radius: Radius for penalty spots
        scale: Scale factor for the pitch
    
    Returns:
        BGR image of the pitch
    """
    import math

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8,
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # Draw edges
    for start, end in config.edges:
        point1 = (
            int(config.vertices[start - 1][0] * scale) + padding,
            int(config.vertices[start - 1][1] * scale) + padding,
        )
        point2 = (
            int(config.vertices[end - 1][0] * scale) + padding,
            int(config.vertices[end - 1][1] * scale) + padding,
        )
        cv2.line(pitch_image, point1, point2, line_color.as_bgr(), line_thickness)

    # Centre circle
    centre_circle_center = (scaled_length // 2 + padding, scaled_width // 2 + padding)
    cv2.circle(pitch_image, centre_circle_center, scaled_circle_radius, line_color.as_bgr(), line_thickness)

    # Penalty spots
    penalty_points = [
        (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
        (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
    ]
    for spot in penalty_points:
        cv2.circle(pitch_image, spot, point_radius, line_color.as_bgr(), -1)

    # Penalty arcs
    offset = (config.penalty_box_length - config.penalty_spot_distance) * scale
    clamped = min(max(offset / max(scaled_circle_radius, 1e-6), -1.0), 1.0)
    arc_angle = math.degrees(math.acos(clamped))

    left_arc_center = (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)
    right_arc_center = (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)

    # Left arc
    cv2.ellipse(
        pitch_image,
        left_arc_center,
        (scaled_circle_radius, scaled_circle_radius),
        0,
        -arc_angle,
        arc_angle,
        line_color.as_bgr(),
        line_thickness,
    )

    # Right arc
    right_start, right_end = 180 - arc_angle, 180 + arc_angle
    cv2.ellipse(
        pitch_image,
        right_arc_center,
        (scaled_circle_radius, scaled_circle_radius),
        0,
        right_start,
        right_end,
        line_color.as_bgr(),
        line_thickness,
    )

    return pitch_image


# ---------------------------------------------------------------------------
# View Composition
# ---------------------------------------------------------------------------

def compose_split_view(
    frame: np.ndarray,
    minimap: np.ndarray,
    min_ratio: float = 0.35,
    stats: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Create a modern dashboard view with frame top and dashboard bottom.
    
    This replaces the old simple stack with a professional layout:
    - Fixed aspect ratio (matches input frame)
    - Dark theme dashboard
    - Proper scaling without black bars
    
    Args:
        frame: Video frame (BGR)
        minimap: Minimap image (BGR)
        min_ratio: Fraction of height dedicated to dashboard (default 0.3)
        stats: Statistics dictionary for display
    
    Returns:
        Composed image
    """
    # Initialize layout constants
    frame_h, frame_w = frame.shape[:2]
    
    # Calculate dashboard sizes
    # We want to keep the frame strictly untouched if possible, or scale it to fit top section
    # Let's create a canvas that maintains width but extends height for the dashboard
    dashboard_h = int(frame_h * 0.35)  # Dashboard is 35% of frame height
    total_h = frame_h + dashboard_h
    total_w = frame_w
    
    # Create canvas (Dark Slate Background: #1a1a1a)
    canvas = np.full((total_h, total_w, 3), (26, 26, 26), dtype=np.uint8)
    
    # --- 1. Place Video Frame (Top) ---
    canvas[:frame_h, :frame_w] = frame
    
    # --- 2. Create Dashboard Area (Bottom) ---
    dashboard_y = frame_h
    
    # Add a subtle separator line
    cv2.line(canvas, (0, dashboard_y), (total_w, dashboard_y), (60, 60, 60), 2)
    
    # --- 3. Place Minimap (Right side of dashboard) ---
    # We want the minimap to fit within the dashboard height with some padding
    mm_padding = 20
    mm_target_h = dashboard_h - (mm_padding * 2)
    
    # Resize minimap to fit height while maintaining aspect ratio
    mm_h, mm_w = minimap.shape[:2]
    mm_scale = mm_target_h / mm_h
    mm_new_w = int(mm_w * mm_scale)
    mm_new_h = int(mm_h * mm_scale)
    
    minimap_resized = cv2.resize(minimap, (mm_new_w, mm_new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate position (Right aligned with padding)
    mm_x = total_w - mm_new_w - mm_padding * 2
    mm_y = dashboard_y + mm_padding
    
    # Draw minimap background/border
    cv2.rectangle(
        canvas, 
        (mm_x - 5, mm_y - 5), 
        (mm_x + mm_new_w + 5, mm_y + mm_new_h + 5), 
        (50, 50, 50), 
        -1
    )
    cv2.rectangle(
        canvas, 
        (mm_x - 5, mm_y - 5), 
        (mm_x + mm_new_w + 5, mm_y + mm_new_h + 5), 
        (100, 100, 100), 
        1
    )
    
    # Blit minimap
    canvas[mm_y:mm_y+mm_new_h, mm_x:mm_x+mm_new_w] = minimap_resized
    
    # --- 4. Draw Statistics (Left side of dashboard) ---
    if stats:
        stats_x = mm_padding * 2
        stats_y = dashboard_y + mm_padding + 30
        line_height = 40
        
        # Helper for text
        def draw_stat_text(img, text, pos, scale=0.8, color=(220, 220, 220), thickness=1):
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)
            
        # Title
        draw_stat_text(canvas, "MATCH ANALYTICS", (stats_x, stats_y - 10), 1.0, (255, 255, 255), 2)
        
        # Stats Columns
        # Col 1: Counts
        col1_x = stats_x
        current_y = stats_y + 40
        
        # Team 0
        t0_count = stats.get('team_0', 0)
        cv2.circle(canvas, (col1_x + 10, current_y - 10), 10, (0, 0, 255), -1) # Red dot
        draw_stat_text(canvas, f"Team A: {t0_count}", (col1_x + 30, current_y))
        
        current_y += line_height
        
        # Team 1
        t1_count = stats.get('team_1', 0)
        cv2.circle(canvas, (col1_x + 10, current_y - 10), 10, (255, 0, 0), -1) # Blue dot
        draw_stat_text(canvas, f"Team B: {t1_count}", (col1_x + 30, current_y))
        
        # Col 2: General Stats
        col2_x = stats_x + 300
        current_y = stats_y + 40
        
        tracked = stats.get('tracked', 0)
        draw_stat_text(canvas, f"Active Players: {tracked}", (col2_x, current_y))
        
        current_y += line_height
        unassigned = stats.get('unassigned', 0)
        draw_stat_text(canvas, f"Unassigned: {unassigned}", (col2_x, current_y), color=(150, 150, 150))

    return canvas


def minimap_coords(
    x: float,
    y: float,
    scale: float = MINIMAP_SCALE,
    padding: int = MINIMAP_PADDING,
) -> Tuple[int, int]:
    """
    Convert pitch coordinates into minimap pixel positions.
    
    Args:
        x: Pitch x coordinate
        y: Pitch y coordinate
        scale: Minimap scale factor
        padding: Minimap padding
    
    Returns:
        (x, y) pixel coordinates on minimap
    """
    return (int(x * scale) + padding, int(y * scale) + padding)


# ---------------------------------------------------------------------------
# Annotator Instances
# ---------------------------------------------------------------------------

def create_annotators(
    config: SoccerPitchConfiguration,
    edge_color: str = '#00BFFF',
    vertex_color: str = '#FF1493',
    detected_vertex_color: str = '#FFD700',
    edge_thickness: int = 2,
    vertex_radius: int = 8,
    detected_radius: int = 6,
) -> Tuple[sv.EdgeAnnotator, sv.VertexAnnotator, sv.VertexAnnotator]:
    """
    Create supervision annotator instances for field visualization.
    
    Args:
        config: Soccer pitch configuration
        edge_color: Color for pitch edges
        vertex_color: Color for pitch vertices
        detected_vertex_color: Color for detected vertices
        edge_thickness: Line thickness for edges
        vertex_radius: Radius for vertex markers
        detected_radius: Radius for detected vertex markers
    
    Returns:
        Tuple of (edge_annotator, vertex_annotator, detected_vertex_annotator)
    """
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex(edge_color),
        thickness=edge_thickness,
        edges=config.line_edges
    )
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex(vertex_color),
        radius=vertex_radius
    )
    detected_vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex(detected_vertex_color),
        radius=detected_radius
    )
    
    return edge_annotator, vertex_annotator, detected_vertex_annotator
