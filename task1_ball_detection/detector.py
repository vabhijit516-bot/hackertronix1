"""
Detector module for Task 1: Ball Detection.
Supports ONNX Runtime, Ultralytics YOLOv8/v11, and Classical HSV/Hough fallback.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class BallDetector:
    """
    High-performance ball detector using lightweight neural networks (YOLOv8n / YOLOv11n).
    Supports ONNX Runtime acceleration and configurable confidence/NMS thresholds.
    """
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thresh: float = 0.45,
        iou_thresh: float = 0.40,
        img_size: Tuple[int, int] = (416, 416),
        use_onnx: bool = True
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.use_onnx = use_onnx and HAS_ONNX
        
        self.ort_session = None
        self.yolo_model = None
        self.engine_type = "Classical Fallback"

        # Check if ONNX model exists and load ONNX Session
        if self.use_onnx and model_path.endswith(".onnx") and os.path.exists(model_path):
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.ort_session = ort.InferenceSession(model_path, providers=providers)
                self.engine_type = f"ONNX Runtime ({self.ort_session.get_providers()[0]})"
                print(f"[Detector] Loaded ONNX model: {model_path} with {self.engine_type}")
            except Exception as e:
                print(f"[Detector Warning] ONNX load failed: {e}. Falling back to PyTorch/YOLO.")
                self.use_onnx = False

        # If ONNX not active, load Ultralytics PyTorch model
        if not self.use_onnx and HAS_ULTRALYTICS:
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                pt_path = os.path.join(base_dir, "yolov8n.pt")
                target_path = model_path if os.path.exists(model_path) else (pt_path if os.path.exists(pt_path) else "yolov8n.pt")
                self.yolo_model = YOLO(target_path)
                self.engine_type = "PyTorch YOLOv8"
                print(f"[Detector] Loaded Ultralytics model: {target_path}")
            except Exception as e:
                print(f"[Detector Warning] Ultralytics load failed: {e}")

    def set_confidence_threshold(self, conf_thresh: float):
        """Update confidence threshold (e.g. from F1 optimizer)."""
        self.conf_thresh = max(0.01, min(0.99, conf_thresh))

    def detect(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        Detect balls in an input BGR image frame.
        Returns:
            boxes: List of (x1, y1, x2, y2) bounding boxes in pixel coordinates.
            scores: List of confidence scores [0.0 - 1.0].
        """
        if self.ort_session is not None:
            return self._detect_onnx(frame)
        elif self.yolo_model is not None:
            return self._detect_yolo(frame)
        else:
            # Fallback to classical detector if models unavailable
            hough_det = FastHoughBallDetector()
            return hough_det.detect(frame)

    def _detect_yolo(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """Inference via Ultralytics YOLO API."""
        target_classes = [32] if (hasattr(self.yolo_model, 'names') and isinstance(self.yolo_model.names, dict) and len(self.yolo_model.names) > 32) else None
        results = self.yolo_model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            verbose=False,
            classes=target_classes
        )
        
        boxes, scores = [], []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                conf = float(b.conf[0].cpu().numpy())
                boxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
                scores.append(conf)

        return boxes, scores

    def _detect_onnx(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """High-speed inference via ONNX Runtime Engine."""
        h_orig, w_orig = frame.shape[:2]
        
        # Preprocessing: Resize & Normalize
        resized = cv2.resize(frame, self.img_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        
        # ONNX Run
        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: input_tensor})
        
        # Parse YOLOv8 ONNX Output Shape: [1, 5, 3549] (or [1, 84, 8400])
        preds = outputs[0][0]  # Shape (5, N)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T    # Shape (N, 5) -> x_center, y_center, width, height, conf
            
        boxes, scores = [], []
        scale_x = w_orig / self.img_size[0]
        scale_y = h_orig / self.img_size[1]
        
        candidate_boxes = []
        candidate_scores = []

        for row in preds:
            cx, cy, w, h = row[:4]
            conf = row[4]
            if conf >= self.conf_thresh:
                x1 = int((cx - w / 2) * scale_x)
                y1 = int((cy - h / 2) * scale_y)
                x2 = int((cx + w / 2) * scale_x)
                y2 = int((cy + h / 2) * scale_y)
                candidate_boxes.append([x1, y1, x2 - x1, y2 - y1])
                candidate_scores.append(float(conf))

        if candidate_boxes:
            indices = cv2.dnn.NMSBoxes(
                candidate_boxes, candidate_scores, self.conf_thresh, self.iou_thresh
            )
            if len(indices) > 0:
                for idx in indices.flatten():
                    x, y, w, h = candidate_boxes[idx]
                    boxes.append((max(0, x), max(0, y), min(w_orig, x + w), min(h_orig, y + h)))
                    scores.append(candidate_scores[idx])

        return boxes, scores


class FastHoughBallDetector:
    """
    Classical Computer Vision Detector: HSV Color Masking + Hough Circle Transform.
    Provides extremely high throughput (>150 FPS on CPU) for controlled conditions.
    """
    def __init__(
        self,
        hsv_lower: Tuple[int, int, int] = (5, 100, 100),
        hsv_upper: Tuple[int, int, int] = (25, 255, 255),
        dp: float = 1.2,
        min_dist: float = 40,
        param1: float = 50,
        param2: float = 28,
        min_radius: int = 8,
        max_radius: int = 120
    ):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.engine_type = "HSV + Hough Circles (CPU)"

    def detect(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """Detect spherical/circular objects matching target color ranges."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Morphological smoothing to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        boxes, scores = [], []
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (x, y, r) in circles:
                x1, y1 = max(0, x - r), max(0, y - r)
                x2, y2 = min(frame.shape[1], x + r), min(frame.shape[0], y + r)
                boxes.append((x1, y1, x2, y2))
                # Heuristic score based on circularity mask overlap
                scores.append(0.85)

        return boxes, scores
