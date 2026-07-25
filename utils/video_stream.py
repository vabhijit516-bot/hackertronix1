"""
Robust Threaded Video Stream Reader with Windows CAP_DSHOW & Synthetic Fallback.
Ensures video stream never hangs or fails even if physical webcam is locked/in-use.
"""

import os
import sys
import time
import cv2
import numpy as np
import threading
from typing import Union, Tuple, Optional


def create_synthetic_demo_video(output_path: str = "synthetic_demo.mp4", duration_sec: int = 10, fps: int = 30):
    """
    Generates a realistic synthetic video feed containing both a bouncing ball
    and a moving human face target for testing monocular vision pipelines.
    """
    if os.path.exists(output_path):
        return output_path

    width, height = 640, 480
    num_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    # Ball motion state
    bx, by = 120.0, 150.0
    bvx, bvy = 7.0, 4.0
    b_radius = 24

    # Face motion state
    fx, fy = 450.0, 200.0
    fvx = -2.5

    for f_idx in range(num_frames):
        # Dark studio background grid
        frame = np.full((height, width, 3), (25, 30, 25), dtype=np.uint8)
        for grid_x in range(0, width, 40):
            cv2.line(frame, (grid_x, 0), (grid_x, height), (35, 42, 35), 1)
        for grid_y in range(0, height, 40):
            cv2.line(frame, (0, grid_y), (width, grid_y), (35, 42, 35), 1)

        # 1. Animate Bouncing Orange Basketball
        bx += bvx
        by += bvy
        bvy += 0.35 # gravity
        if bx - b_radius <= 20 or bx + b_radius >= width - 20:
            bvx *= -0.95
            bx = np.clip(bx, b_radius + 20, width - b_radius - 20)
        if by + b_radius >= height - 30:
            bvy *= -0.85
            by = height - b_radius - 30

        bcx, bcy = int(bx), int(by)
        cv2.circle(frame, (bcx, bcy), b_radius, (0, 140, 255), -1)
        cv2.circle(frame, (bcx, bcy), b_radius, (0, 90, 200), 2)

        # 2. Animate Moving Human Face Simulation
        fx += fvx
        if fx <= 200 or fx >= 520:
            fvx *= -1.0
            
        fcx, fcy = int(fx), int(fy + 15 * np.sin(f_idx * 0.1))
        # Draw face oval (Skin tone)
        cv2.ellipse(frame, (fcx, fcy), (55, 75), 0, 0, 360, (160, 195, 230), -1)
        # Eyes
        cv2.circle(frame, (fcx - 20, fcy - 15), 6, (40, 40, 40), -1)
        cv2.circle(frame, (fcx + 20, fcy - 15), 6, (40, 40, 40), -1)
        # Mouth
        cv2.ellipse(frame, (fcx, fcy + 25), (18, 8), 0, 0, 180, (60, 60, 180), 2)

        out.write(frame)

    out.release()
    print(f"[VideoStream] Generated synthetic fallback video: {output_path}")
    return output_path


class ThreadedVideoStream:
    """
    Asynchronous threaded video stream reader.
    Tries physical webcam devices first (with DSHOW backend), then falls back to synthetic video loop.
    """
    def __init__(self, src: Union[int, str] = 0, name: str = "ThreadedVideoStream"):
        self.src = src
        self.name = name
        self.stopped = False
        self.lock = threading.Lock()
        self.grabbed = False
        self.frame = None
        self.using_synthetic = False
        self.last_reconnect_attempt = time.time()

        self._open_hardware_or_synthetic()

    def _open_hardware_or_synthetic(self):
        """Attempt to open physical hardware camera, or set up synthetic fallback."""
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.using_synthetic = False

        if isinstance(self.src, int) or (isinstance(self.src, str) and str(self.src).isdigit()):
            cam_idx = int(self.src)
            if sys.platform.startswith("win"):
                self.stream = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if self.stream is None or not self.stream.isOpened():
                self.stream = cv2.VideoCapture(cam_idx)
        else:
            self.stream = cv2.VideoCapture(self.src)

        if self.stream is not None and self.stream.isOpened():
            for _ in range(10):
                self.grabbed, self.frame = self.stream.read()
                if self.grabbed and self.frame is not None:
                    print(f"[VideoStream] Physical webcam initialized successfully (resolution: {self.frame.shape[1]}x{self.frame.shape[0]}).")
                    break
                time.sleep(0.05)

        if not self.grabbed or self.frame is None:
            print("[VideoStream Warning] Physical webcam unavailable or locked. Falling back to Synthetic Demo Video.")
            self.using_synthetic = True
            synthetic_path = create_synthetic_demo_video()
            if self.stream is not None:
                self.stream.release()
            self.stream = cv2.VideoCapture(synthetic_path)
            self.grabbed, self.frame = self.stream.read()

    def start(self):
        """Start background capture thread."""
        t = threading.Thread(target=self.update, name=self.name, daemon=True)
        t.start()
        return self

    def update(self):
        """Continuously grab frames from stream with automatic hardware camera reconnect."""
        while not self.stopped:
            # If using synthetic fallback, periodically try reconnecting to physical webcam
            if self.using_synthetic and (time.time() - self.last_reconnect_attempt > 4.0):
                self.last_reconnect_attempt = time.time()
                try:
                    cam_idx = int(self.src) if (isinstance(self.src, int) or (isinstance(self.src, str) and str(self.src).isdigit())) else 0
                    test_cam = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(cam_idx)
                    if test_cam.isOpened():
                        ret, test_frame = test_cam.read()
                        if ret and test_frame is not None:
                            print("[VideoStream] Physical webcam re-connected! Switching back to live hardware camera stream.")
                            with self.lock:
                                if self.stream is not None:
                                    self.stream.release()
                                self.stream = test_cam
                                self.using_synthetic = False
                                self.grabbed = ret
                                self.frame = test_frame
                            continue
                    test_cam.release()
                except Exception:
                    pass

            if self.stream is None or not self.stream.isOpened():
                time.sleep(0.01)
                continue

            grabbed, frame = self.stream.read()

            if not grabbed or frame is None:
                if self.using_synthetic:
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    grabbed, frame = self.stream.read()
                else:
                    time.sleep(0.01)
                    continue

            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

            time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the latest frame."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        """Stop thread and release stream."""
        self.stopped = True
        if self.stream is not None and self.stream.isOpened():
            self.stream.release()
