"""
Global configuration settings for Computer Vision Qualifier tasks.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List

# Base workspace path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class BallDetectionConfig:
    """Configuration for Task 1: Ball Detection & Tracking."""
    # Model parameters
    model_path: str = os.path.join(BASE_DIR, "models", "yolov8n_ball.onnx")
    fallback_model_pt: str = "yolov8n.pt"  # Pre-trained fallback
    img_size: Tuple[int, int] = (416, 416)
    conf_threshold: float = 0.45           # Tuned via optimizer for max F1
    nms_threshold: float = 0.40
    use_onnx: bool = True
    
    # Tracking parameters
    enable_tracking: bool = True
    detect_interval: int = 3               # Run heavy detector every N frames
    max_cosine_distance: float = 0.2
    nn_budget: int = 100
    max_iou_distance: float = 0.7
    max_age: int = 30
    n_init: int = 3
    
    # Fallback HSV Color Thresholding parameters (for legacy/extreme fast mode)
    hsv_lower: Tuple[int, int, int] = (5, 100, 100)   # Default: Orange ball
    hsv_upper: Tuple[int, int, int] = (15, 255, 255)
    dp: float = 1.2
    min_dist: float = 50
    param1: float = 50
    param2: float = 30
    min_radius: int = 5
    max_radius: int = 150

@dataclass
class FaceDistanceConfig:
    """Configuration for Task 2: Face Distance & Angle Estimation."""
    # Camera Intrinsic Parameters
    focal_length_px: float = 850.0          # Calibrated focal length in pixels
    real_face_width_m: float = 0.15          # Average human face width in meters (15 cm)
    inter_ocular_width_m: float = 0.063      # Average distance between pupil centers (6.3 cm)
    
    # MediaPipe settings
    min_detection_confidence: float = 0.6
    use_landmarks: bool = True               # Use landmark distance (eyes/cheeks) for stable width
    
    # Kalman Filter / Smoothing settings
    filter_type: str = "kalman"              # 'kalman', 'ema', or 'none'
    ema_alpha: float = 0.35                  # Smoothing factor for Exponential Moving Average
    kalman_process_noise: float = 1e-4
    kalman_measurement_noise: float = 1e-2

@dataclass
class AppConfig:
    """Overall Application Configuration."""
    task1: BallDetectionConfig = field(default_factory=BallDetectionConfig)
    task2: FaceDistanceConfig = field(default_factory=FaceDistanceConfig)
    
    # Stream settings
    default_camera_id: int = 0
    display_width: int = 1280
    display_height: int = 720
    target_fps: int = 60
