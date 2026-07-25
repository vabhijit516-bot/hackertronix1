"""
Training and ONNX Exporter utility for Task 1 (Ball Detection).
Exports trained PyTorch YOLO models to ONNX format for accelerated inference.
"""

import os
import sys
import argparse

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def train_ball_model(
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 416,
    batch_size: int = 32,
    model_name: str = "yolov8n.pt"
):
    """
    Train lightweight YOLOv8n model with heavy domain augmentations
    (motion blur, lighting jitter, scale, occlusion) for real-world ball detection.
    """
    if not HAS_ULTRALYTICS:
        print("[Error] Ultralytics package is required for training. Install with: pip install ultralytics")
        return None

    print(f"=== Starting YOLO Training ({model_name}) ===")
    model = YOLO(model_name)
    
    # Train with augmentations suited for ball tracking
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        hsv_h=0.015,     # Hue augmentation
        hsv_s=0.7,       # Saturation jitter
        hsv_v=0.4,       # Value/brightness jitter
        scale=0.5,       # Scale variation
        perspective=0.0005,
        blur=0.1,        # Simulate fast motion blur
        mosaic=1.0,      # Combine context images
        mixup=0.1,
        project="runs_ball",
        name="ball_detector_nano"
    )
    
    best_weights = os.path.join("runs_ball", "ball_detector_nano", "weights", "best.pt")
    print(f"[Success] Best model saved to: {best_weights}")
    return best_weights


def export_to_onnx(model_path: str, imgsz: int = 416, half_precision: bool = False) -> str:
    """
    Export PyTorch model weights (.pt) to optimized ONNX format (.onnx).
    """
    if not HAS_ULTRALYTICS:
        print("[Error] Ultralytics package is required for ONNX export.")
        return ""

    if not os.path.exists(model_path):
        print(f"[Error] Model path {model_path} does not exist.")
        return ""

    print(f"=== Exporting {model_path} to ONNX (imgsz={imgsz}) ===")
    model = YOLO(model_path)
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half_precision,
        simplify=True,
        dynamic=False
    )
    print(f"[Success] ONNX Model exported to: {onnx_path}")
    return str(onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1 Model Train & Export Script")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to input PyTorch model (.pt)")
    parser.add_argument("--export", action="store_true", help="Export existing model to ONNX format")
    parser.add_argument("--imgsz", type=int, default=416, help="Input resolution (416 or 512 for FPS optimization)")
    args = parser.parse_args()

    if args.export:
        export_to_onnx(args.model, imgsz=args.imgsz)
    else:
        print("Usage: python train_export.py --model yolov8n.pt --export --imgsz 416")
