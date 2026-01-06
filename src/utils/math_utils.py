"""
Utility functions for mathematical operations.
"""
import numpy as np

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector to unit length (L2 norm).

    Args:
        v (np.ndarray): Feature vector

    Returns:
        np.ndarray: L2 normalized vector
    """
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    Box format: [x1, y1, x2, y2]
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

def check_occlusions(tracker_results_list, iou_threshold=0.15):
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

    # Compare every pair
    n = len(tracker_results_list)
    for i in range(n):
        for j in range(i + 1, n):
            box1 = tracker_results_list[i][0]
            tid1 = tracker_results_list[i][1]
            
            box2 = tracker_results_list[j][0]
            tid2 = tracker_results_list[j][1]
            
            iou = calculate_iou(box1, box2)
            
            # Distance based check (in case they are close but IoU is low, e.g. one above another)
            # Center distance
            c1 = ((box1[0]+box1[2])/2, (box1[1]+box1[3])/2)
            c2 = ((box2[0]+box2[2])/2, (box2[1]+box2[3])/2)
            dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
            
            # Heuristic: If centers are closer than 1/3 of average width, it's risky
            avg_width = ((box1[2]-box1[0]) + (box2[2]-box2[0])) / 2
            
            if iou > iou_threshold or dist < (avg_width * 0.8):
                occlusion_map[tid1] = True
                occlusion_map[tid2] = True
                
    return occlusion_map
