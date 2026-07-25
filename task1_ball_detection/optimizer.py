"""
F1 Threshold Optimizer for Task 1.
Sweeps confidence thresholds to find the operating point that maximizes F1 score.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
try:
    from .tracker import compute_iou
except ImportError:
    from task1_ball_detection.tracker import compute_iou


class F1Optimizer:
    """
    Evaluates detector predictions against ground truth bounding boxes across
    confidence thresholds to compute Precision, Recall, and max F1 score.
    """
    def __init__(self, iou_match_thresh: float = 0.5):
        self.iou_match_thresh = iou_match_thresh

    def evaluate_predictions(
        self,
        all_pred_boxes: List[List[Tuple[int, int, int, int]]],
        all_pred_scores: List[List[float]],
        all_gt_boxes: List[List[Tuple[int, int, int, int]]],
        conf_thresholds: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute Precision, Recall, and F1 curve over confidence thresholds.
        
        Args:
            all_pred_boxes: Bounding box predictions per image.
            all_pred_scores: Confidence scores per prediction.
            all_gt_boxes: Ground truth bounding boxes per image.
            conf_thresholds: Array of thresholds to evaluate.
        """
        if conf_thresholds is None:
            conf_thresholds = np.linspace(0.05, 0.95, 19)

        precisions = []
        recalls = []
        f1_scores = []

        total_gt = sum(len(gt) for gt in all_gt_boxes)
        if total_gt == 0:
            print("[Optimizer Warning] Total Ground Truth count is zero.")
            return {"best_conf": 0.5, "best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0}

        for conf in conf_thresholds:
            tp, fp, fn = 0, 0, 0
            
            for pred_b, pred_s, gt_b in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
                # Filter predictions by confidence
                valid_preds = [b for b, s in zip(pred_b, pred_s) if s >= conf]
                gt_matched = [False] * len(gt_b)
                
                for pb in valid_preds:
                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, gb in enumerate(gt_b):
                        if not gt_matched[g_idx]:
                            iou = compute_iou(pb, gb)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx
                                
                    if best_iou >= self.iou_match_thresh and best_gt_idx != -1:
                        tp += 1
                        gt_matched[best_gt_idx] = True
                    else:
                        fp += 1
                        
                fn += sum(1 for matched in gt_matched if not matched)

            precision = tp / float(tp + fp + 1e-6)
            recall = tp / float(total_gt + 1e-6)
            f1 = 2 * precision * recall / float(precision + recall + 1e-6)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        best_idx = int(np.argmax(f1_scores))
        best_conf = float(conf_thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])

        return {
            "conf_thresholds": conf_thresholds,
            "precisions": precisions,
            "recalls": recalls,
            "f1_scores": f1_scores,
            "best_conf": best_conf,
            "best_f1": best_f1,
            "best_precision": precisions[best_idx],
            "best_recall": recalls[best_idx]
        }

    def plot_pr_f1_curve(self, results: Dict[str, Any], save_path: Optional[str] = "f1_optimization_curve.png"):
        """Plot Precision, Recall, and F1 curves as a function of confidence threshold."""
        plt.figure(figsize=(9, 5))
        confs = results["conf_thresholds"]
        
        plt.plot(confs, results["precisions"], 'b--', label='Precision', linewidth=2)
        plt.plot(confs, results["recalls"], 'g--', label='Recall', linewidth=2)
        plt.plot(confs, results["f1_scores"], 'r-', label='F1 Score', linewidth=2.5)
        
        best_conf = results["best_conf"]
        best_f1 = results["best_f1"]
        plt.axvline(x=best_conf, color='gray', linestyle=':', label=f'Optimal Conf={best_conf:.2f} (F1={best_f1:.3f})')
        
        plt.title('Task 1: Ball Detection Threshold vs. Precision / Recall / F1')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Metric Value')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left')

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"[Optimizer] F1 curve plot saved to {save_path}")
        plt.close()
