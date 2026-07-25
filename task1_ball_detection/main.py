"""
Main application runner for Task 1: Ball Detection & Tracking.
Executes real-time webcam/video stream with interactive controls and performance HUD.
"""

import sys
import os
import time
import cv2
import argparse

# Add parent workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import BallDetectionConfig
from utils.video_stream import ThreadedVideoStream
from utils.visualization import draw_ball_detection, draw_fps_hud
from task1_ball_detection.detector import BallDetector, FastHoughBallDetector
from task1_ball_detection.tracker import BallTracker


def main():
    parser = argparse.ArgumentParser(description="Task 1 Real-Time Ball Detection & Tracking App")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (0) or video file path")
    parser.add_argument("--onnx", action="store_true", help="Force ONNX Runtime engine")
    parser.add_argument("--hough", action="store_true", help="Use Hough Circle fallback detector")
    parser.add_argument("--skip", type=int, default=3, help="Run heavy detector every N frames")
    args = parser.parse_args()

    # Parse input source
    src = int(args.source) if args.source.isdigit() else args.source
    
    cfg = BallDetectionConfig()
    
    # Initialize Detector
    if args.hough:
        detector = FastHoughBallDetector()
    else:
        detector = BallDetector(model_path=cfg.model_path, use_onnx=args.onnx or cfg.use_onnx)
        
    tracker = BallTracker(max_age=cfg.max_age)
    
    print(f"=== Starting Task 1: Ball Detection ===")
    print(f"Engine: {detector.engine_type}")
    print(f"Inter-frame Skipping: N={args.skip}")
    print("Press 'q' to quit | Press 'm' to toggle detector mode | Press '+' / '-' to adjust confidence threshold")

    # Start Video Stream
    video_stream = ThreadedVideoStream(src=src).start()
    time.sleep(0.5)

    frame_count = 0
    fps = 0.0
    fps_start_time = time.perf_counter()
    fps_frame_counter = 0

    cached_boxes = []
    cached_scores = []
    cached_track_ids = []

    try:
        while True:
            grabbed, frame = video_stream.read()
            if not grabbed or frame is None:
                # Loop video if source is a video file
                if isinstance(src, str) and os.path.exists(src):
                    video_stream.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[Info] End of video stream.")
                    break

            # Process frame
            if frame_count % args.skip == 0:
                boxes, scores = detector.detect(frame)
                
                if cfg.enable_tracking:
                    tracked_results = tracker.update(boxes)
                    cached_boxes = [b for b, trk_id in tracked_results]
                    cached_track_ids = [trk_id for b, trk_id in tracked_results]
                    cached_scores = scores if len(scores) == len(cached_boxes) else [0.9] * len(cached_boxes)
                else:
                    cached_boxes, cached_scores = boxes, scores
                    cached_track_ids = []
            else:
                # Inter-frame tracking state
                if cfg.enable_tracking and len(tracker.trackers) > 0:
                    tracked_results = tracker.step_interframe()
                    cached_boxes = [b for b, trk_id in tracked_results]
                    cached_track_ids = [trk_id for b, trk_id in tracked_results]
                    cached_scores = [0.85] * len(cached_boxes)

            # Draw output visualizations
            frame = draw_ball_detection(
                frame,
                cached_boxes,
                cached_scores,
                track_ids=cached_track_ids,
                is_tracked=(frame_count % args.skip != 0)
            )

            # Measure FPS
            fps_frame_counter += 1
            if fps_frame_counter >= 10:
                now = time.perf_counter()
                fps = fps_frame_counter / (now - fps_start_time + 1e-6)
                fps_start_time = now
                fps_frame_counter = 0

            # Draw HUD
            hud_info = f"Conf: {getattr(detector, 'conf_thresh', 0.45):.2f}"
            frame = draw_fps_hud(frame, fps, method_name=detector.engine_type, extra_info=hud_info)

            cv2.imshow("Task 1 - Ball Detection & Tracking", frame)
            frame_count += 1

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                if hasattr(detector, 'set_confidence_threshold'):
                    detector.set_confidence_threshold(detector.conf_thresh + 0.05)
                    print(f"[Controls] Confidence threshold set to: {detector.conf_thresh:.2f}")
            elif key == ord('-'):
                if hasattr(detector, 'set_confidence_threshold'):
                    detector.set_confidence_threshold(detector.conf_thresh - 0.05)
                    print(f"[Controls] Confidence threshold set to: {detector.conf_thresh:.2f}")

    finally:
        video_stream.stop()
        cv2.destroyAllWindows()
        print("[Task 1] Application closed clean.")

if __name__ == "__main__":
    main()
