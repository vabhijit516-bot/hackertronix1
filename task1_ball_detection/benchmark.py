"""
Benchmark suite for Task 1 (Ball Detection & Tracking).
Measures F1 score, Precision, Recall, Inference Latency, and Pipeline FPS.
"""

import time
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from task1_ball_detection.detector import BallDetector, FastHoughBallDetector
from task1_ball_detection.tracker import BallTracker
from task1_ball_detection.optimizer import F1Optimizer



def generate_synthetic_ball_video(
    output_path: str = "synthetic_ball_bench.mp4",
    num_frames: int = 150,
    width: int = 640,
    height: int = 480
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Generates a synthetic bouncing ball video sequence with ground truth bounding boxes.
    Simulates motion blur, scaling, and lighting variations.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    # Bouncing ball physics parameters
    x, y = 100.0, 100.0
    vx, vy = 8.0, 5.0
    radius = 25
    gravity = 0.4
    
    gt_boxes_per_frame = []

    for frame_idx in range(num_frames):
        # Create realistic background with noise
        frame = np.full((height, width, 3), (40, 50, 40), dtype=np.uint8)
        
        # Physics update
        x += vx
        y += vy
        vy += gravity
        
        # Bounce off walls
        if x - radius <= 0 or x + radius >= width:
            vx *= -0.95
            x = np.clip(x, radius, width - radius)
        if y + radius >= height:
            vy *= -0.85
            y = height - radius
            
        cx, cy = int(x), int(y)
        r = int(radius + 3 * np.sin(frame_idx * 0.1)) # slight scale variation
        
        # Draw bouncing ball (orange basketball style)
        cv2.circle(frame, (cx, cy), r, (0, 140, 255), -1)
        cv2.circle(frame, (cx, cy), r, (0, 90, 200), 2)
        cv2.line(frame, (cx - r, cy), (cx + r, cy), (0, 50, 150), 2)
        
        # Add light motion blur
        if abs(vy) > 8:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            
        out.write(frame)
        
        # Save Ground Truth bounding box
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(width, cx + r), min(height, cy + r)
        gt_boxes_per_frame.append([(x1, y1, x2, y2)])

    out.release()
    print(f"[Benchmark] Generated synthetic evaluation video: {output_path}")
    return gt_boxes_per_frame


def run_benchmark(video_path: str = "synthetic_ball_bench.mp4", gt_boxes: Optional[List] = None):
    """Run full benchmark evaluating FPS vs F1 Score across pipelines."""
    print("\n=======================================================")
    print("       TASK 1: BALL DETECTION BENCHMARK EVALUATOR      ")
    print("=======================================================\n")

    if gt_boxes is None or not os.path.exists(video_path):
        gt_boxes = generate_synthetic_ball_video(video_path)

    # Test Configurations
    configs = [
        {"name": "HSV/Hough Classical (CPU)", "detector": FastHoughBallDetector(), "use_tracker": False, "skip": 1},
        {"name": "HSV + Inter-frame Tracker (Skipping N=3)", "detector": FastHoughBallDetector(), "use_tracker": True, "skip": 3},
        {"name": "YOLO/ONNX Standard Engine", "detector": BallDetector(use_onnx=True), "use_tracker": False, "skip": 1},
        {"name": "YOLO/ONNX + Kalman Tracker (Skipping N=3)", "detector": BallDetector(use_onnx=True), "use_tracker": True, "skip": 3},
    ]

    optimizer = F1Optimizer(iou_match_thresh=0.4)

    for cfg in configs:
        cap = cv2.VideoCapture(video_path)
        detector = cfg["detector"]
        use_tracker = cfg["use_tracker"]
        skip = cfg["skip"]

        all_pred_boxes = []
        all_pred_scores = []
        
        frame_idx = 0
        tracker = BallTracker() if use_tracker else None
        cached_boxes, cached_scores = [], []
        
        start_time = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            if frame_idx % skip == 0:
                boxes, scores = detector.detect(frame)
                if use_tracker:
                    tracked = tracker.update(boxes)
                    cached_boxes = [b for b, trk_id in tracked]
                    cached_scores = [0.9] * len(cached_boxes)
                else:
                    cached_boxes, cached_scores = boxes, scores
            else:
                if use_tracker and tracker is not None:
                    # Update trackers via Kalman prediction
                    tracked = tracker.step_interframe()
                    cached_boxes = [b for b, trk_id in tracked]
                    cached_scores = [0.85] * len(cached_boxes)

            all_pred_boxes.append(cached_boxes)
            all_pred_scores.append(cached_scores)
            frame_idx += 1

        elapsed = time.perf_counter() - start_time
        cap.release()
        
        fps = frame_idx / max(elapsed, 1e-5)
        
        # Evaluate F1 score on aligned frames
        min_len = min(len(all_pred_boxes), len(gt_boxes))
        opt_res = optimizer.evaluate_predictions(
            all_pred_boxes[:min_len],
            all_pred_scores[:min_len],
            gt_boxes[:min_len]
        )

        print(f"Pipeline Mode : {cfg['name']}")
        print(f"  └─ Processed : {frame_idx} frames in {elapsed:.3f}s")
        print(f"  └─ Speed     : {fps:.1f} FPS")
        print(f"  └─ Precision : {opt_res['best_precision']:.3f}")
        print(f"  └─ Recall    : {opt_res['best_recall']:.3f}")
        print(f"  └─ Max F1    : {opt_res['best_f1']:.3f} (Conf Thresh={opt_res['best_conf']:.2f})")
        print(f"  └─ Combined  : F1 * FPS = {opt_res['best_f1'] * fps:.1f}\n")

if __name__ == "__main__":
    import os
    run_benchmark()
