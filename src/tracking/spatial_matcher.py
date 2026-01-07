"""
Spatial matching for re-acquiring lost persons using motion prediction.
"""
import numpy as np
from utils.config import Config


class SpatialMatcher:
    """
    Handles spatial re-acquisition of temporarily lost persons
    using IoU, distance, and time decay metrics.
    """

    def __init__(self):
        """Initialize spatial matcher."""
        self.short_term_memory_ttl = Config.SHORT_TERM_MEMORY_TTL
        print("Spatial matcher initialized.")

    def get_box_center(self, box: list) -> tuple[int, int]:
        """
        Calculate center coordinates of a bounding box.

        Args:
            box (list): [x1, y1, x2, y2]

        Returns:
            tuple: (center_x, center_y)
        """
        return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

    def calculate_iou(self, box1: list, box2: list) -> float:
        """
        Calculate Intersection over Union (IoU).

        Args:
            box1 (list): [x1, y1, x2, y2]
            box2 (list): [x1, y1, x2, y2]

        Returns:
            float: IoU score (0.0 to 1.0)
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

    def predict_next_position(self, last_box: list, time_elapsed: float) -> list:
        """
        Predict next position based on time elapsed.

        Args:
            last_box (list): [x1, y1, x2, y2]
            time_elapsed (float): Seconds since last seen

        Returns:
            list: Predicted box [x1, y1, x2, y2]
        """
        expansion_factor = min(Config.SPATIAL_PREDICTION_FACTOR_1 * time_elapsed, Config.SPATIAL_PREDICTION_FACTOR_2)
        x1, y1, x2, y2 = last_box

        return [
            x1 - expansion_factor,
            y1 - expansion_factor,
            x2 + expansion_factor,
            y2 + expansion_factor
        ]

    def adaptive_spatial_threshold_sq(self, bbox: list) -> float:
        """
        Calculate SQUARED adaptive distance threshold based on bbox size.
        Avoids sqrt calculation.

        Args:
            bbox (list): [x1, y1, x2, y2]

        Returns:
            float: Squared Adaptive threshold
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        # bbox_diag_sq = width^2 + height^2
        bbox_diag_sq = bbox_width ** 2 + bbox_height ** 2

        # threshold = diag * factor -> threshold_sq = diag_sq * factor^2
        threshold_sq = min(bbox_diag_sq * (Config.ADAPTIVE_THRESHOLD_FACTOR ** 2), Config.ADAPTIVE_THRESHOLD_MAX ** 2)
        threshold_sq = max(threshold_sq, Config.ADAPTIVE_THRESHOLD_MIN ** 2)

        return threshold_sq

    def find_spatial_match(self, new_box: list, lost_pid_buffer: dict, current_time: float) -> tuple[int | None, float]:
        """
        Find best spatial match from lost PIDs.

        Args:
            new_box (list): Current detection box [x1, y1, x2, y2]
            lost_pid_buffer (dict): Buffer of lost PIDs
            current_time (float): Current timestamp

        Returns:
            tuple: (matched_pid: int or None, match_score: float)
        """
        new_box_center = self.get_box_center(new_box)
        best_match_pid = None
        best_match_score = -1.0
        
        # Pre-calculate adaptive threshold squared once
        adaptive_thresh_sq = self.adaptive_spatial_threshold_sq(new_box)
        adaptive_thresh = np.sqrt(adaptive_thresh_sq) # Needed for dist_score normalization only

        for pid, data in lost_pid_buffer.items():
            time_elapsed = current_time - data["disappear_time"]
            last_box = data["last_box"]

            # Motion prediction
            predicted_box = self.predict_next_position(last_box, time_elapsed)

            # IoU calculation
            iou = self.calculate_iou(new_box, predicted_box)

            # Center distance (Squared)
            lost_box_center = self.get_box_center(last_box)
            center_dist_sq = (new_box_center[0] - lost_box_center[0])**2 + (new_box_center[1] - lost_box_center[1])**2
            center_dist = np.sqrt(center_dist_sq)

            # Time decay
            time_factor = np.exp(-time_elapsed / self.short_term_memory_ttl)

            # Combined score
            iou_score = iou * 2.0
            dist_score = max(0, 1.0 - (center_dist / adaptive_thresh))
            time_score = time_factor

            match_score = (iou_score * Config.SPATIAL_IOU_WEIGHT + dist_score * Config.SPATIAL_DIST_WEIGHT + time_score * Config.SPATIAL_TIME_WEIGHT)

            if match_score > Config.SPATIAL_MATCH_MIN_SCORE and match_score > best_match_score:
                best_match_score = match_score
                best_match_pid = pid

        return best_match_pid, best_match_score