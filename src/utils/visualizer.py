"""
Visualization utilities for drawing HUD and bounding boxes.
"""
import cv2
from utils.config import Config


class Visualizer:
    """Handles all visualization tasks (HUD, bounding boxes, labels)."""

    @staticmethod
    def draw_hud(frame, fps, detection_threshold, match_threshold, db_size, tta_enabled):
        """
        Draws Head-Up Display (HUD) on frame.

        Args:
            frame (np.ndarray): Video frame
            fps (float): Current FPS
            detection_threshold (float): Detection threshold
            match_threshold (float): Match threshold
            db_size (int): Number of enrolled persons
            tta_enabled (bool): TTA status
        """
        tta_indicator = "TTA:ON" if tta_enabled else "TTA:OFF"

        hud_text1 = f"FPS: {fps:.1f} | Detection (U/J): {detection_threshold:.2f}"
        hud_text2 = f"Match (Y/H): {match_threshold:.2f} | DB: {db_size} | {tta_indicator}"

        # Background rectangle
        cv2.rectangle(frame, (5, 5), (500, 60), (0, 0, 0), -1)

        # Text
        cv2.putText(frame, hud_text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, hud_text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    @staticmethod
    def draw_person_bbox(frame, bbox, label, color):
        """
        Draws bounding box and label for a person.

        Args:
            frame (np.ndarray): Video frame
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            label (str): Text label
            color (tuple): BGR color
        """
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)