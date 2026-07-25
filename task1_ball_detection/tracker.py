"""
Tracking engine for Task 1: Inter-frame Kalman Filter & IoU Tracking.
Allows running the heavy neural network detector every N frames while tracking on intermediate frames.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional

class SingleBallKalmanTracker:
    """
    Constant-Velocity Kalman Filter for tracking a single ball bounding box (x, y, dx, dy, w, h).
    """
    def __init__(self, bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        w, h = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))
        cx, cy = float(x1 + w / 2.0), float(y1 + h / 2.0)
        
        # State vector [cx, cy, dx, dy, w, h]
        self.kf = cv2.KalmanFilter(6, 4)
        
        # Measurement matrix H [4x6] -> measures [cx, cy, w, h]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Transition matrix F [6x6]
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0,  0, 0],
            [0, 1, 0,  dt, 0, 0],
            [0, 0, 1,  0,  0, 0],
            [0, 0, 0,  1,  0, 0],
            [0, 0, 0,  0,  1, 0],
            [0, 0, 0,  0,  0, 1]
        ], dtype=np.float32)
        
        # Noise covariance matrices
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[2, 2] *= 5.0  # Velocity variance
        self.kf.processNoiseCov[3, 3] *= 5.0
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0

        # Initial state
        self.kf.statePost = np.array([[cx], [cy], [0.0], [0.0], [w], [h]], dtype=np.float32)
        self.current_bbox = bbox
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1

    def predict(self) -> Tuple[int, int, int, int]:
        """Advance state prediction step."""
        prediction = self.kf.predict()
        cx, cy = float(prediction[0, 0]), float(prediction[1, 0])
        w, h = max(1.0, float(prediction[4, 0])), max(1.0, float(prediction[5, 0]))

        x1 = int(cx - w / 2.0)
        y1 = int(cy - h / 2.0)
        x2 = int(cx + w / 2.0)
        y2 = int(cy + h / 2.0)
        
        self.current_bbox = (x1, y1, x2, y2)
        self.time_since_update += 1
        return self.current_bbox

    def update(self, bbox: Tuple[int, int, int, int]):
        """Correction step with new ground-truth detector measurement."""
        x1, y1, x2, y2 = bbox
        w, h = float(x2 - x1), float(y2 - y1)
        cx, cy = float(x1 + w / 2.0), float(y1 + h / 2.0)
        
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)
        self.current_bbox = bbox
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self) -> Tuple[int, int, int, int]:
        """Return current estimated bounding box without advancing prediction state."""
        return self.current_bbox


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


class BallTracker:
    """
    Multi-object tracker for ball trajectories using IoU association and Kalman Filters.
    Enables detector-skipping frame strategies for maximized FPS.
    """
    def __init__(self, max_age: int = 30, iou_thresh: float = 0.3):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.trackers: Dict[int, SingleBallKalmanTracker] = {}
        self.next_id = 1

    def update(self, detected_boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[Tuple[int, int, int, int], int]]:
        """
        Update tracker state with new detection frame.
        Returns list of ((x1, y1, x2, y2), track_id).
        """
        # Predict all active trackers (ONCE per frame)
        predicted_boxes = {}
        for trk_id, trk in list(self.trackers.items()):
            predicted_boxes[trk_id] = trk.predict()
            
        # Match detections to predicted boxes via greedy IoU matching
        unmatched_detections = set(range(len(detected_boxes)))
        unmatched_trackers = set(self.trackers.keys())
        
        matches = []
        if len(detected_boxes) > 0 and len(predicted_boxes) > 0:
            iou_matrix = np.zeros((len(detected_boxes), len(predicted_boxes)), dtype=np.float32)
            trk_ids = list(predicted_boxes.keys())
            
            for i, det_box in enumerate(detected_boxes):
                for j, trk_id in enumerate(trk_ids):
                    iou_matrix[i, j] = compute_iou(det_box, predicted_boxes[trk_id])
                    
            while True:
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_val = iou_matrix[max_idx]
                if max_val < self.iou_thresh:
                    break
                det_idx, trk_idx = max_idx[0], max_idx[1]
                trk_id = trk_ids[trk_idx]
                matches.append((det_idx, trk_id))
                
                iou_matrix[det_idx, :] = -1.0
                iou_matrix[:, trk_idx] = -1.0
                unmatched_detections.discard(det_idx)
                unmatched_trackers.discard(trk_id)

        # Update matched trackers
        for det_idx, trk_id in matches:
            self.trackers[trk_id].update(detected_boxes[det_idx])
            
        # Create new trackers for unmatched detections (avoid ID spam and duplicate tracks)
        for det_idx in unmatched_detections:
            det_box = detected_boxes[det_idx]
            # Check if detection overlaps heavily with any existing active tracker
            has_overlap = any(compute_iou(det_box, trk.get_state()) > 0.4 for trk in self.trackers.values())
            if not has_overlap and len(self.trackers) < 5:
                self.trackers[self.next_id] = SingleBallKalmanTracker(det_box)
                self.next_id += 1

        # Purge stale trackers exceeding max_age or unconfirmed 1-hit noise
        for trk_id in list(self.trackers.keys()):
            trk = self.trackers[trk_id]
            if trk.time_since_update > self.max_age:
                del self.trackers[trk_id]
            elif trk.hits < 2 and trk.time_since_update > 0:
                # Instantly purge unconfirmed 1-hit noise detections
                del self.trackers[trk_id]

        # Track Deduplication (Track NMS): prune overlapping redundant trackers
        active_ids = list(self.trackers.keys())
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                id1, id2 = active_ids[i], active_ids[j]
                if id1 in self.trackers and id2 in self.trackers:
                    box1 = self.trackers[id1].get_state()
                    box2 = self.trackers[id2].get_state()
                    if compute_iou(box1, box2) > 0.4:
                        t1, t2 = self.trackers[id1], self.trackers[id2]
                        if t1.hits >= t2.hits:
                            del self.trackers[id2]
                        else:
                            del self.trackers[id1]
                
        # Output active bounding boxes and IDs (only confirmed tracks with hits >= 2 or freshly updated)
        active_results = []
        for trk_id, trk in self.trackers.items():
            if trk.hits >= 2:
                active_results.append((trk.get_state(), trk_id))
                
        return active_results

    def step_interframe(self) -> List[Tuple[Tuple[int, int, int, int], int]]:
        """
        Inter-frame prediction step when detector is skipped.
        Advances Kalman state once and returns active valid tracks.
        """
        for trk_id, trk in list(self.trackers.items()):
            trk.predict()
            if trk.time_since_update > self.max_age:
                del self.trackers[trk_id]
            elif trk.hits < 2 and trk.time_since_update > 0:
                del self.trackers[trk_id]
                
        active_results = []
        for trk_id, trk in self.trackers.items():
            if trk.hits >= 2 and trk.time_since_update <= 4:
                active_results.append((trk.get_state(), trk_id))
                
        return active_results
