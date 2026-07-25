"""
Unified Auto-Detection Computer Vision Pipeline.
Combines Task 1 (Ball Detection & Inter-frame Tracking) and Task 2 (Monocular Face Distance & Angle Estimation)
into a single real-time multi-task processing engine.
"""

import os
import sys
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import BallDetectionConfig, FaceDistanceConfig
from task1_ball_detection.detector import BallDetector, FastHoughBallDetector
from task1_ball_detection.tracker import BallTracker
from task2_face_distance.estimator import MonocularFaceEstimator
from utils.visualization import draw_ball_detection, draw_face_metrics, draw_fps_hud


class UnifiedVisionPipeline:
    """
    Unified real-time multi-task engine that automatically detects and tracks balls
    and estimates 3D monocular face distances simultaneously on every frame.
    """
    def __init__(self, t1_cfg: Optional[BallDetectionConfig] = None, t2_cfg: Optional[FaceDistanceConfig] = None):
        self.t1_cfg = t1_cfg if t1_cfg is not None else BallDetectionConfig()
        self.t2_cfg = t2_cfg if t2_cfg is not None else FaceDistanceConfig()

        # Task 1 Engine
        self.ball_detector = BallDetector(
            model_path=self.t1_cfg.model_path,
            conf_thresh=self.t1_cfg.conf_threshold,
            use_onnx=self.t1_cfg.use_onnx
        )
        self.ball_tracker = BallTracker(max_age=self.t1_cfg.max_age)

        # Task 2 Engine
        self.face_estimator = MonocularFaceEstimator(
            focal_length_px=self.t2_cfg.focal_length_px,
            real_face_width_m=self.t2_cfg.real_face_width_m,
            inter_ocular_width_m=self.t2_cfg.inter_ocular_width_m,
            use_landmarks=self.t2_cfg.use_landmarks,
            filter_type=self.t2_cfg.filter_type
        )

        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes an input video frame for both Ball Detection and Face Distance simultaneously.
        
        Returns:
            processed_frame: BGR frame with visual overlays.
            telemetry: Dict containing counts, detection states, and 3D position metrics.
        """
        # 1. Process Task 1: Ball Detection & Tracking
        cached_boxes, cached_scores, cached_ids = [], [], []
        
        if self.frame_count % self.t1_cfg.detect_interval == 0:
            boxes, scores = self.ball_detector.detect(frame)
            if self.t1_cfg.enable_tracking:
                tracked_results = self.ball_tracker.update(boxes)
                cached_boxes = [b for b, trk_id in tracked_results]
                cached_ids = [trk_id for b, trk_id in tracked_results]
                cached_scores = scores if len(scores) == len(cached_boxes) else [0.9] * len(cached_boxes)
            else:
                cached_boxes, cached_scores = boxes, scores
                cached_ids = []
        else:
            if self.t1_cfg.enable_tracking and len(self.ball_tracker.trackers) > 0:
                cached_boxes = [trk.predict() for trk in self.ball_tracker.trackers.values()]
                cached_ids = list(self.ball_tracker.trackers.keys())
                cached_scores = [0.85] * len(cached_boxes)

        # Draw ball detections if present
        if len(cached_boxes) > 0:
            frame = draw_ball_detection(
                frame, cached_boxes, cached_scores, track_ids=cached_ids,
                is_tracked=(self.frame_count % self.t1_cfg.detect_interval != 0)
            )

        # 2. Process Task 2: Monocular Face Distance & Angle Estimation
        face_results = self.face_estimator.process_frame(frame)
        
        # Draw face 3D depth & angle metrics if present
        for face in face_results:
            frame = draw_face_metrics(
                frame, bbox=face['bbox'], distance_m=face['distance_m'],
                angle_deg=face['angle_deg'], landmarks=face['landmarks']
            )

        # 3. Compile Combined Telemetry
        ball_detected = len(cached_boxes) > 0
        face_detected = len(face_results) > 0

        last_z = round(face_results[0]['distance_m'], 2) if face_detected else 0.0
        last_angle = round(face_results[0]['angle_deg'], 1) if face_detected else 0.0

        telemetry = {
            "ball_detected": ball_detected,
            "balls_count": len(cached_boxes),
            "face_detected": face_detected,
            "faces_count": len(face_results),
            "last_z": last_z,
            "last_angle": last_angle,
            "engine": self.ball_detector.engine_type,
            "focal_px": self.face_estimator.f_px
        }

        # Status Overlay Tag at top right
        status_str = "AUTO-DETECT: "
        if ball_detected and face_detected:
            status_str += f"⚽ {len(cached_boxes)} Ball(s) | 👤 Z={last_z:.2f}m"
        elif ball_detected:
            status_str += f"⚽ {len(cached_boxes)} Ball(s) Tracked"
        elif face_detected:
            status_str += f"👤 Face Depth Z={last_z:.2f}m ({last_angle:+.1f}°)"
        else:
            status_str += "Scanning for Objects..."

        cv2.putText(frame, status_str, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        self.frame_count += 1
        return frame, telemetry
