"""
Professional High-End Visualization Engine for Computer Vision HUD Overlays.
Designed with modern Cyberpunk/Automotive AI aesthetics, corner brackets, and crisp typography.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

# Premium Color Palette (BGR)
COLOR_NEON_CYAN = (254, 242, 0)     # #00f2fe (Cyan/Blue glow)
COLOR_NEON_GREEN = (118, 230, 0)   # #00e676 (Green glow)
COLOR_NEON_ORANGE = (0, 145, 255)   # #ff9100 (Orange glow)
COLOR_NEON_PURPLE = (255, 64, 129)  # #ff4081 (Pink/Purple)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GLASS = (15, 12, 8)      # Deep dark glass background
COLOR_ACCENT_BORDER = (80, 70, 50)


def draw_corner_brackets(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    line_len: int = 15
) -> np.ndarray:
    """Draw stylish L-shaped corner brackets on bounding boxes for futuristic AI styling."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, thickness)

    return img


def draw_fps_hud(frame: np.ndarray, fps: float, method_name: str = "", extra_info: str = "") -> np.ndarray:
    """
    Draw a professional, sleek translucent HUD panel in the top-left of the video frame.
    Guarantees no text collision or top-edge clipping.
    """
    h_orig, w_orig, _ = frame.shape
    
    fps_str = f"FPS {fps:.1f}"
    engine_str = f"MODEL: {method_name.upper()}" if method_name else ""
    
    y_start = 22
    box_h = 72
    
    # Calculate box width
    (fps_w, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
    (eng_w, _), _ = cv2.getTextSize(engine_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    (ext_w, _), _ = cv2.getTextSize(extra_info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    
    box_w = max(450, fps_w + ext_w + 90, eng_w + 50)
    
    # Translucent Dark Glass Backdrop
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, y_start), (20 + box_w, y_start + box_h), COLOR_DARK_GLASS, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    
    # Neon Accent Outer Border
    cv2.rectangle(frame, (20, y_start), (20 + box_w, y_start + box_h), COLOR_NEON_CYAN, 1, cv2.LINE_AA)
    
    # Left accent vertical indicator bar
    cv2.rectangle(frame, (20, y_start), (24, y_start + box_h), COLOR_NEON_CYAN, -1)

    # Row 1: FPS Gauge (Green) + Target Telemetry (Orange)
    cv2.putText(frame, fps_str, (36, y_start + 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_NEON_GREEN, 2, cv2.LINE_AA)
    
    if extra_info:
        # Subtle divider pipe
        cv2.line(frame, (160, y_start + 12), (160, y_start + 32), COLOR_ACCENT_BORDER, 1)
        cv2.putText(frame, extra_info, (175, y_start + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEON_ORANGE, 1, cv2.LINE_AA)
        
    # Row 2: Model Architecture Engine Tag (White)
    if method_name:
        cv2.putText(frame, engine_str, (36, y_start + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)
        
    return frame


def draw_ball_detection(
    frame: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    track_ids: Optional[List[int]] = None,
    is_tracked: bool = False
) -> np.ndarray:
    """Render professional bounding boxes, tracking brackets, and confidence badges for detected balls."""
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        score = scores[i] if i < len(scores) else 0.0
        color = COLOR_NEON_CYAN if not is_tracked else COLOR_NEON_GREEN
        
        # Semi-transparent box interior highlight
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        
        # Main Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        
        # Corner brackets for high-end look
        draw_corner_brackets(frame, (x1, y1), (x2, y2), color, thickness=2, line_len=12)
        
        # Center Target Crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 3, COLOR_NEON_CYAN, -1)
        cv2.line(frame, (cx - 8, cy), (cx + 8, cy), COLOR_NEON_CYAN, 1)
        cv2.line(frame, (cx, cy - 8), (cx, cy + 8), COLOR_NEON_CYAN, 1)
        
        # Label Badge
        label = f"BALL {score:.2f}"
        if track_ids and i < len(track_ids):
            label = f"TRACK #{track_ids[i]} ({score:.2f})"
            
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_y1 = max(y1 - h_text - 8, 10)
        
        # Badge background
        cv2.rectangle(frame, (x1, badge_y1), (x1 + w_text + 12, badge_y1 + h_text + 6), COLOR_DARK_GLASS, -1)
        cv2.rectangle(frame, (x1, badge_y1), (x1 + w_text + 12, badge_y1 + h_text + 6), color, 1)
        cv2.putText(frame, label, (x1 + 6, badge_y1 + h_text + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
        
    return frame


def draw_face_metrics(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    distance_m: float,
    angle_deg: float,
    landmarks: Optional[List[Tuple[int, int]]] = None
) -> np.ndarray:
    """Render professional 3D position metrics (Depth Z & Angle theta) for faces."""
    x1, y1, x2, y2 = bbox
    h_img, w_img, _ = frame.shape
    c_x, c_y = w_img // 2, h_img // 2
    
    # Semi-transparent face box interior
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_NEON_GREEN, -1)
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)
    
    # Face Bounding Box & Corner Brackets
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NEON_GREEN, 1, cv2.LINE_AA)
    draw_corner_brackets(frame, (x1, y1), (x2, y2), COLOR_NEON_GREEN, thickness=2, line_len=15)
    
    # Target Center Dot
    fc_x, fc_y = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (fc_x, fc_y), 4, COLOR_NEON_ORANGE, -1)
    
    # Ray line connecting camera optical center to face center
    cv2.line(frame, (c_x, c_y), (fc_x, fc_y), (100, 200, 100), 1, cv2.LINE_AA)
    
    # Facial Landmark dots
    if landmarks:
        for (lx, ly) in landmarks:
            cv2.circle(frame, (lx, ly), 2, COLOR_NEON_GREEN, -1)
            
    # Metric Readout Panel Pill above face
    metrics_str = f"DEPTH Z: {distance_m:.2f}m  |  ANGLE: {angle_deg:+.1f} deg"
    (w_t, h_t), _ = cv2.getTextSize(metrics_str, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    
    panel_y1 = max(y1 - h_t - 10, 15)
    cv2.rectangle(frame, (x1, panel_y1), (x1 + w_t + 16, panel_y1 + h_t + 8), COLOR_DARK_GLASS, -1)
    cv2.rectangle(frame, (x1, panel_y1), (x1 + w_t + 16, panel_y1 + h_t + 8), COLOR_NEON_GREEN, 1)
    cv2.putText(frame, metrics_str, (x1 + 8, panel_y1 + h_t + 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    
    # Subtle Optical Crosshair at image center
    cv2.line(frame, (c_x - 12, c_y), (c_x + 12, c_y), (80, 80, 80), 1)
    cv2.line(frame, (c_x, c_y - 12), (c_x, c_y + 12), (80, 80, 80), 1)
    
    return frame
