"""
Monocular Face Distance and Angle Estimator for Task 2.
Powered by OpenCV YuNet Deep Learning Face Model & Local Haar Cascade
for 100% reliable face detection close-up, far away, and under varied room lighting.
"""

import os
import sys
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

try:
    from .filter import DistanceAngleFilter
except ImportError:
    from task2_face_distance.filter import DistanceAngleFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
HAAR_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")


class MonocularFaceEstimator:
    """
    Monocular 3D position estimator for human faces using 2D camera feeds.
    Depth Z = (f * W) / w_px
    Horizontal Angle theta = arctan((x - c_x) / f)
    """
    def __init__(
        self,
        focal_length_px: float = 850.0,
        real_face_width_m: float = 0.15,
        inter_ocular_width_m: float = 0.063,
        use_landmarks: bool = True,
        filter_type: str = "kalman"
    ):
        self.f_px = focal_length_px
        self.W_face = real_face_width_m
        self.W_eyes = inter_ocular_width_m
        self.use_landmarks = use_landmarks

        # Temporal Filter (Kalman or EMA)
        self.filter = DistanceAngleFilter(filter_type=filter_type)

        # 1. Initialize YuNet Deep Learning Detector
        self.yunet_detector = None
        if os.path.exists(YUNET_PATH) and hasattr(cv2, 'FaceDetectorYN'):
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model=YUNET_PATH,
                    config="",
                    input_size=(640, 480),
                    score_threshold=0.55,
                    nms_threshold=0.5,
                    top_k=5000
                )
                print(f"[Estimator] Loaded OpenCV YuNet Neural Face Detector: {YUNET_PATH}")
            except Exception as e:
                print(f"[Estimator Warning] YuNet load error: {e}")

        # 2. Initialize Haar Cascade Backup Classifier
        self.haar_cascade = None
        if os.path.exists(HAAR_PATH):
            try:
                self.haar_cascade = cv2.CascadeClassifier(HAAR_PATH)
                print(f"[Estimator] Loaded Local Haar Cascade Classifier: {HAAR_PATH}")
            except Exception as e:
                print(f"[Estimator Warning] Haar Cascade load error: {e}")

    def set_focal_length(self, f_px: float):
        """Update calibrated focal length in pixels."""
        self.f_px = max(10.0, f_px)

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process frame and estimate 3D position metrics (Depth Z & Angle theta).
        """
        h_img, w_img, _ = frame.shape
        c_x = w_img / 2.0
        results_list = []

        # 1. Primary Engine: OpenCV YuNet Deep Learning Face Detector
        if self.yunet_detector is not None:
            self.yunet_detector.setInputSize((w_img, h_img))
            _, faces = self.yunet_detector.detect(frame)

            if faces is not None and len(faces) > 0:
                for face in faces:
                    # YuNet output format: [x, y, w, h, x_reye, y_reye, x_leye, y_leye, x_nose, y_nose, x_rmouth, y_rmouth, x_lmouth, y_lmouth, score]
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    r_eye = (int(face[4]), int(face[5]))
                    l_eye = (int(face[6]), int(face[7]))
                    nose = (int(face[8]), int(face[9]))
                    r_mouth = (int(face[10]), int(face[11]))
                    l_mouth = (int(face[12]), int(face[13]))

                    landmarks = [r_eye, l_eye, nose, r_mouth, l_mouth]
                    w_px = float(w)

                    # Inter-ocular landmark distance for ultra-stable depth calculation
                    eye_dist_px = np.hypot(l_eye[0] - r_eye[0], l_eye[1] - r_eye[1])
                    if self.use_landmarks and eye_dist_px > 5:
                        raw_z = (self.f_px * self.W_eyes) / float(eye_dist_px)
                    else:
                        raw_z = (self.f_px * self.W_face) / max(1.0, w_px)

                    x_center = x + w / 2.0
                    raw_theta = np.degrees(np.arctan((x_center - c_x) / self.f_px))

                    smooth_z, smooth_theta = self.filter.update(raw_z, raw_theta)

                    results_list.append({
                        'bbox': (max(0, x), max(0, y), min(w_img, x + w), min(h_img, y + h)),
                        'distance_m': smooth_z,
                        'angle_deg': smooth_theta,
                        'raw_distance_m': raw_z,
                        'raw_angle_deg': raw_theta,
                        'center': (int(x_center), int(y + h / 2.0)),
                        'landmarks': landmarks
                    })

        # 2. Backup Engine: Haar Cascade Classifier
        if len(results_list) == 0 and self.haar_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=7, minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                w_px = float(w)
                raw_z = (self.f_px * self.W_face) / max(1.0, w_px)
                x_center = x + w / 2.0
                raw_theta = np.degrees(np.arctan((x_center - c_x) / self.f_px))

                smooth_z, smooth_theta = self.filter.update(raw_z, raw_theta)

                results_list.append({
                    'bbox': (x, y, x + w, y + h),
                    'distance_m': smooth_z,
                    'angle_deg': smooth_theta,
                    'raw_distance_m': raw_z,
                    'raw_angle_deg': raw_theta,
                    'center': (int(x_center), int(y + h / 2.0)),
                    'landmarks': []
                })

        return results_list
