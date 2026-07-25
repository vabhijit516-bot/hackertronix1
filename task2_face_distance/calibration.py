"""
Camera Calibration module for Task 2 (Monocular Face Distance Estimation).
Supports OpenCV Checkerboard Calibration and Quick Reference-Object Pinhole Calibration.
"""

import os
import glob
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


def reference_object_calibration(
    w_ref_px: float,
    z_ref_m: float,
    w_real_m: float = 0.15
) -> float:
    """
    Computes camera focal length f in pixel units using a reference object of known size.
    
    Formula: f_px = (w_ref_px * Z_ref_m) / W_real_m
    
    Args:
        w_ref_px: Width of the reference object in pixels as measured on screen.
        z_ref_m: Known distance from camera lens to the reference object (in meters).
        w_real_m: Known real physical width of the reference object (in meters, e.g. 0.15m for face).
        
    Returns:
        Calibrated focal length in pixels (f_px).
    """
    if w_real_m <= 0 or w_ref_px <= 0:
        raise ValueError("Width and pixel measurements must be positive non-zero numbers.")
    f_px = (w_ref_px * z_ref_m) / w_real_m
    return float(f_px)


class CameraCalibrator:
    """
    Computes camera intrinsic matrix K and distortion coefficients D
    from a set of checkerboard calibration images using OpenCV cv2.calibrateCamera.
    """
    def __init__(self, checkerboard_size: Tuple[int, int] = (9, 6), square_size_m: float = 0.025):
        self.checkerboard_size = checkerboard_size
        self.square_size_m = square_size_m
        
        # Prepare 3D object points (0,0,0), (1,0,0), (2,0,0) ...
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_m
        
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.focal_length_px: float = 850.0

    def calibrate_from_images(self, image_folder: str, file_extension: str = "jpg") -> Dict[str, Any]:
        """Runs pinhole calibration over a set of checkerboard images."""
        objpoints = [] # 3D point in real world space
        imgpoints = [] # 2D points in image plane
        
        images = glob.glob(os.path.join(image_folder, f"*.{file_extension}"))
        if not images:
            print(f"[Calibration Error] No images found in {image_folder}")
            return {}

        img_shape = None
        found_count = 0

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_shape = gray.shape[::-1]
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                found_count += 1
                objpoints.append(self.objp)
                # Refine corner locations
                criteria = (cv2.TERMCRITERIA_EPS + cv2.TERMCRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        if found_count == 0 or img_shape is None:
            print("[Calibration Error] Checkerboard pattern could not be found in images.")
            return {}

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
        
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.focal_length_px = float((mtx[0, 0] + mtx[1, 1]) / 2.0)
        
        print(f"[Calibration Success] Found pattern in {found_count}/{len(images)} images.")
        print(f"  └─ Focal Length f_x: {mtx[0, 0]:.2f} px | f_y: {mtx[1, 1]:.2f} px")
        print(f"  └─ Principal Center (c_x, c_y): ({mtx[0, 2]:.1f}, {mtx[1, 2]:.1f})")

        return {
            "reprojection_error": ret,
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "focal_length_px": self.focal_length_px
        }

if __name__ == "__main__":
    # Interactive quick calibration test
    w_px = 250.0  # Measured 250 pixels on screen
    z_m = 0.51    # Measured 51 cm away
    w_face = 0.15 # Real face width = 15 cm
    f_calib = reference_object_calibration(w_px, z_m, w_face)
    print(f"Quick Reference Calibration Result: f = {f_calib:.2f} px")
