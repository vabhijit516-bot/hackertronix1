"""
Flask Web Application Server for Monocular Computer Vision Qualifier.
Serves a Unified Auto-Detect Real-Time Video Feed & REST API on http://localhost:5000.
"""

import os
import sys
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from unified_pipeline import UnifiedVisionPipeline
from task1_ball_detection.detector import BallDetector, FastHoughBallDetector
from task2_face_distance.calibration import reference_object_calibration
from utils.visualization import draw_fps_hud
from utils.video_stream import ThreadedVideoStream

app = Flask(__name__)

# Initialize Unified Auto-Detect Pipeline
pipeline = UnifiedVisionPipeline()

# Shared Video Reader Stream
camera_stream = None

def get_camera_stream():
    global camera_stream
    if camera_stream is None or camera_stream.stopped:
        camera_stream = ThreadedVideoStream(src=0).start()
        time.sleep(0.3)
    return camera_stream

# Global Telemetry Cache
telemetry_cache = {
    "fps": 0.0,
    "ball_detected": False,
    "balls_count": 0,
    "face_detected": False,
    "faces_count": 0,
    "last_z": 0.0,
    "last_angle": 0.0,
    "engine": "YOLOv8 ONNX",
    "focal_px": 850.0,
    "filter": "kalman",
    "conf_thresh": 0.45
}


def generate_unified_frames():
    """MJPEG video generator for single unified auto-detect stream."""
    stream = get_camera_stream()
    fps = 0.0
    fps_start = time.perf_counter()
    fps_counter = 0

    while True:
        grabbed, frame = stream.read()
        if not grabbed or frame is None:
            # Fallback frame if webcam unavailable
            frame = np.full((480, 640, 3), (30, 30, 30), dtype=np.uint8)
            cv2.putText(frame, "Webcam feed loading or unavailable", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            time.sleep(0.05)

        # Execute Unified Auto-Detect Pipeline
        frame, telem = pipeline.process_frame(frame)

        # Measure FPS
        fps_counter += 1
        if fps_counter >= 10:
            now = time.perf_counter()
            fps = fps_counter / (now - fps_start + 1e-6)
            fps_start = now
            fps_counter = 0

        # Update telemetry
        telem["fps"] = round(fps, 1)
        telemetry_cache.update(telem)

        # Draw HUD header
        extra_hud = f"Balls: {telem['balls_count']} | Faces: {telem['faces_count']}"
        frame = draw_fps_hud(frame, fps, method_name=telem["engine"], extra_info=extra_hud)

        # Encode JPEG stream
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.01)


# Flask Routes
@app.route('/')
def index():
    """Unified Dashboard."""
    return render_template('index.html')


@app.route('/video_feed/unified')
def video_feed_unified():
    """Single Unified MJPEG Video Stream."""
    return Response(generate_unified_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns live unified telemetry."""
    return jsonify(telemetry_cache)


@app.route('/api/config', methods=['POST'])
def update_config():
    """Unified API to adjust detector engine, threshold, focal length, or filter mode."""
    data = request.json or {}

    if 'conf_thresh' in data:
        conf = float(data['conf_thresh'])
        pipeline.ball_detector.set_confidence_threshold(conf)
        telemetry_cache["conf_thresh"] = conf

    if 'engine' in data:
        engine_mode = data['engine']
        if engine_mode == "hough":
            pipeline.ball_detector = FastHoughBallDetector()
        else:
            pipeline.ball_detector = BallDetector(model_path=pipeline.t1_cfg.model_path, use_onnx=True)

    if 'focal_px' in data:
        f_px = float(data['focal_px'])
        pipeline.face_estimator.set_focal_length(f_px)

    if 'filter_type' in data:
        f_type = str(data['filter_type']).lower()
        pipeline.face_estimator.filter.filter_type = f_type
        telemetry_cache["filter"] = f_type

    return jsonify({"status": "success"})


@app.route('/api/calibrate', methods=['POST'])
def run_quick_calibration():
    """API endpoint for reference object focal solver."""
    data = request.json or {}
    w_px = float(data.get('w_px', 250.0))
    z_m = float(data.get('z_m', 0.5))
    w_real_m = float(data.get('w_real_m', 0.15))

    try:
        new_f = reference_object_calibration(w_px, z_m, w_real_m)
        pipeline.face_estimator.set_focal_length(new_f)
        telemetry_cache["focal_px"] = new_f
        return jsonify({"status": "success", "focal_px": new_f})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    print("\n=========================================================")
    print("  🚀 Unified Auto-Detect CV Server Running!")
    print("  👉 Dashboard URL: http://localhost:5000")
    print("=========================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
