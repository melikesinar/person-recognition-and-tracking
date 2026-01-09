"""
Utility functions for mathematical operations.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Union

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector to unit length (L2 norm).
    Safe against zero division.

    Args:
        v (np.ndarray): Feature vector (1D) or matrix of vectors (2D)

    Returns:
        np.ndarray: L2 normalized vector/matrix
    """
    v = np.asarray(v, dtype=np.float32)
    
    # Handle 1D case
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v if n == 0 else v / n
    
    # Handle 2D case (N x D)
    # Calculate norm along axis 1, keep dimensions for broadcasting
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    
    # Avoid division by zero
    norm[norm == 0] = 1.0 
    
    return v / norm

def calculate_iou(box1: Union[List[float], np.ndarray], box2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    Box format: [x1, y1, x2, y2]
    
    Args:
        box1: First bounding box
        box2: Second bounding box
        
    Returns:
        float: IoU value (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

def check_occlusions(tracker_results_list: List[Any], iou_threshold: float = 0.15) -> Dict[int, bool]:
    """
    Checks for occlusions among all detected boxes in the current frame.
    
    Args:
        tracker_results_list: List of (box, track_id, conf, cls, ind, crop)
        iou_threshold: IoU threshold to consider as occlusion.
    
    Returns:
        dict: {track_id: bool} (True if occluded)
    """
    occlusion_map = {}
    
    # Initialize all as False
    for res in tracker_results_list:
        track_id = res[1]
        occlusion_map[track_id] = False

    n = len(tracker_results_list)
    if n < 2:
        return occlusion_map

    # Extract boxes and IDs for faster processing
    # Assuming box format in tracker_results_list is correct
    boxes = np.array([res[0] for res in tracker_results_list])
    ids = [res[1] for res in tracker_results_list]
    
    # Calculate centers
    # box: x1, y1, x2, y2
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    widths = boxes[:, 2] - boxes[:, 0]
    
    # Compare every pair
    for i in range(n):
        for j in range(i + 1, n):
            # 1. Check IoU
            iou = calculate_iou(boxes[i], boxes[j])
            
            # 2. Check Center Distance
            dist_sq = (centers_x[i] - centers_x[j])**2 + (centers_y[i] - centers_y[j])**2
            dist = np.sqrt(dist_sq)
            
            # Heuristic: If centers are closer than 80% of average width
            avg_width = (widths[i] + widths[j]) / 2
            
            if iou > iou_threshold or dist < (avg_width * 0.8):
                occlusion_map[ids[i]] = True
                occlusion_map[ids[j]] = True
                
    return occlusion_map