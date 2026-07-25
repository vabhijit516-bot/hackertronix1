"""
Temporal Smoothing and Filtering for Task 2 (Distance and Angle).
Provides 1D Kalman Filter and Exponential Moving Average (EMA) to suppress bounding box jitter.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class Kalman1DFilter:
    """1D Linear Kalman Filter for scalar signal smoothing (Depth Z or Angle theta)."""
    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-1):
        self.kf = cv2.KalmanFilter(2, 1)
        
        # State vector: [value, velocity]
        self.kf.measurementMatrix = np.array([[1.0, 0.0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(2, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.array([[measurement_noise]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 1.0
        
        self.initialized = False

    def update(self, measurement: float) -> float:
        """Prediction & Correction step for new 1D scalar measurement."""
        if not self.initialized:
            self.kf.statePost = np.array([[measurement], [0.0]], dtype=np.float32)
            self.initialized = True
            return measurement
            
        self.kf.predict()
        meas_arr = np.array([[measurement]], dtype=np.float32)
        self.kf.correct(meas_arr)
        
        filtered_val = float(self.kf.statePost[0, 0])
        return filtered_val


class DistanceAngleFilter:
    """
    Combined filter manager for (Distance Z, Angle theta) metrics.
    Supports 'kalman', 'ema', and 'none'.
    """
    def __init__(self, filter_type: str = "kalman", ema_alpha: float = 0.35):
        self.filter_type = filter_type.lower()
        self.ema_alpha = ema_alpha
        
        # Kalman Filters
        self.z_kalman = Kalman1DFilter(process_noise=1e-4, measurement_noise=1e-2)
        self.theta_kalman = Kalman1DFilter(process_noise=1e-3, measurement_noise=1e-1)
        
        # EMA States
        self.ema_z: Optional[float] = None
        self.ema_theta: Optional[float] = None

    def update(self, raw_z: float, raw_theta: float) -> Tuple[float, float]:
        """Apply selected temporal filter to raw depth Z and angle theta."""
        if self.filter_type == "kalman":
            z_smooth = self.z_kalman.update(raw_z)
            theta_smooth = self.theta_kalman.update(raw_theta)
            return z_smooth, theta_smooth

        elif self.filter_type == "ema":
            if self.ema_z is None:
                self.ema_z = raw_z
                self.ema_theta = raw_theta
            else:
                self.ema_z = self.ema_alpha * raw_z + (1 - self.ema_alpha) * self.ema_z
                self.ema_theta = self.ema_alpha * raw_theta + (1 - self.ema_alpha) * self.ema_theta
            return self.ema_z, self.ema_theta

        else:
            # Pass-through un-filtered
            return raw_z, raw_theta
