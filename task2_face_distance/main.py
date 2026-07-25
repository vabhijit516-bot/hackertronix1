"""
Main application runner for Task 2: Monocular Face Distance & Angle Estimation.
Executes real-time webcam/video stream processing with interactive controls and 3D overlay HUD.
"""

import sys
import os
import time
import cv2
import argparse

# Add parent workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import FaceDistanceConfig
from utils.video_stream import ThreadedVideoStream
from utils.visualization import draw_face_metrics, draw_fps_hud
from task2_face_distance.estimator import MonocularFaceEstimator
from task2_face_distance.calibration import reference_object_calibration


def main():
    parser = argparse.ArgumentParser(description="Task 2 Monocular Face Distance & Angle Estimator")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (0) or video file path")
    parser.add_argument("--focal", type=float, default=850.0, help="Calibrated camera focal length in pixels")
    parser.add_argument("--filter", type=str, default="kalman", choices=["kalman", "ema", "none"], help="Temporal filter type")
    args = parser.parse_args()

    # Parse input source
    src = int(args.source) if args.source.isdigit() else args.source

    cfg = FaceDistanceConfig()
    estimator = MonocularFaceEstimator(
        focal_length_px=args.focal if args.focal > 0 else cfg.focal_length_px,
        real_face_width_m=cfg.real_face_width_m,
        inter_ocular_width_m=cfg.inter_ocular_width_m,
        use_landmarks=cfg.use_landmarks,
        filter_type=args.filter
    )

    print("=== Starting Task 2: Monocular Face Distance & Angle Estimation ===")
    print(f"Initial Focal Length (f): {estimator.f_px:.1f} px")
    print(f"Real Face Width Assumed : {estimator.W_face * 100:.1f} cm")
    print(f"Filter Type             : {args.filter.upper()}")
    print("Press 'q' to quit | Press 'c' to run quick focal calibration | Press '+' / '-' to adjust focal length")

    # Start Threaded Stream
    video_stream = ThreadedVideoStream(src=src).start()
    time.sleep(0.5)

    fps = 0.0
    fps_start_time = time.perf_counter()
    fps_frame_counter = 0

    try:
        while True:
            grabbed, frame = video_stream.read()
            if not grabbed or frame is None:
                if isinstance(src, str) and os.path.exists(src):
                    video_stream.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[Info] End of video stream.")
                    break

            # Process face distance & angle estimation
            results = estimator.process_frame(frame)

            # Draw visual metrics overlay
            for face in results:
                frame = draw_face_metrics(
                    frame,
                    bbox=face['bbox'],
                    distance_m=face['distance_m'],
                    angle_deg=face['angle_deg'],
                    landmarks=face['landmarks']
                )

            # FPS counter
            fps_frame_counter += 1
            if fps_frame_counter >= 10:
                now = time.perf_counter()
                fps = fps_frame_counter / (now - fps_start_time + 1e-6)
                fps_start_time = now
                fps_frame_counter = 0

            # Draw HUD
            hud_info = f"f={estimator.f_px:.0f}px | Filter:{args.filter.upper()}"
            frame = draw_fps_hud(frame, fps, method_name="Monocular Pinhole Model", extra_info=hud_info)

            cv2.imshow("Task 2 - Monocular Face Distance & Angle", frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                estimator.set_focal_length(estimator.f_px + 25.0)
                print(f"[Controls] Focal length set to: {estimator.f_px:.1f} px")
            elif key == ord('-'):
                estimator.set_focal_length(estimator.f_px - 25.0)
                print(f"[Controls] Focal length set to: {estimator.f_px:.1f} px")
            elif key == ord('c'):
                print("\n--- Quick Reference Calibration Prompt ---")
                try:
                    w_ref_input = float(input("Enter face width measured in pixels (w_px): "))
                    z_ref_input = float(input("Enter actual distance in meters (Z_ref): "))
                    new_f = reference_object_calibration(w_ref_input, z_ref_input, estimator.W_face)
                    estimator.set_focal_length(new_f)
                    print(f"[Calibration] Updated focal length: f = {new_f:.1f} px\n")
                except Exception as e:
                    print(f"[Calibration Error] Invalid input: {e}\n")

    finally:
        video_stream.stop()
        cv2.destroyAllWindows()
        print("[Task 2] Application closed clean.")

if __name__ == "__main__":
    main()
