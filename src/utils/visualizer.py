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
        Minimalist version: Only shows FPS with transparent background.
        """
        hud_text = f"FPS: {fps:.1f}"

        # Text only (No background rectangle)
        cv2.putText(frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    @staticmethod
    def draw_person_bbox(frame, bbox, label, color):
        """
        Draws bounding box and label for a person.
        Label is placed BELOW the feet.

        Args:
            frame (np.ndarray): Video frame
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            label (str): Text label
            color (tuple): BGR color
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label BELOW the feet (y2 + offset)
        # We ensure it doesn't go off-screen if possible, but basic logic is y2+20
        label_y = min(y2 + 25, frame.shape[0] - 10) 
        
        cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)