"""
Visualization utilities for drawing HUD and bounding boxes.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from utils.config import Config

class Visualizer:
    """Handles all visualization tasks (HUD, bounding boxes, labels)."""

    # UI Colors (BGR)
    COLOR_TEXT = (255, 255, 255)       # White
    COLOR_TEXT_BG = (0, 0, 0)          # Black
    COLOR_HUD = (0, 255, 0)            # Green

    @staticmethod
    def draw_hud(frame: np.ndarray, 
                 fps: float, 
                 detection_threshold: float, 
                 match_threshold: float, 
                 db_size: int, 
                 tta_enabled: bool) -> None:
        """
        Draws Head-Up Display (HUD) on frame.
        
        Args:
            frame: Video frame (modified in-place)
            fps: Current Frames Per Second
            detection_threshold: Current detection confidence threshold
            match_threshold: Current ReID matching threshold
            db_size: Current number of identities in database
            tta_enabled: Whether Test-Time Augmentation is active
        """
        # 1. FPS (Top Left, Green)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Visualizer.COLOR_HUD, 2)

        # 2. System Stats (Top Left, below FPS, smaller, White)
        # We only show this if it's meaningful (e.g. not 0)
        info_text = f"DB: {db_size} IDs | TTA: {'ON' if tta_enabled else 'OFF'}"
        cv2.putText(frame, info_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 3. Thresholds (Bottom Left, small, Grey) - Optional, good for debugging
        thresh_text = f"Det: {detection_threshold:.2f} | Match: {match_threshold:.2f}"
        h, w = frame.shape[:2]
        cv2.putText(frame, thresh_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    @staticmethod
    def draw_person_bbox(frame: np.ndarray, 
                         bbox: Tuple[int, int, int, int], 
                         label: str, 
                         color: Tuple[int, int, int],
                         conf: Optional[float] = None) -> None:
        """
        Draws bounding box and label for a person with a text background for readability.
        Label is placed BELOW the feet.

        Args:
            frame (np.ndarray): Video frame
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            label (str): Text label (e.g., "ID: 42")
            color (tuple): BGR color for box and text background
            conf (float, optional): Confidence score to display
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # 1. Draw Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 2. Prepare Label
        full_label = label
        if conf is not None:
            full_label += f" {conf:.2f}"
            
        # 3. Calculate Text Size & Position
        font_scale = 0.6
        font_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Position: Below feet (y2), centered horizontally on box if possible, or left-aligned
        # Let's align left of the box for consistency
        text_x = x1
        text_y = y2 + 25
        
        # Ensure text doesn't go off bottom of screen
        frame_h = frame.shape[0]
        if text_y + 5 > frame_h:
            text_y = y2 - 10 # Flip to inside/above if at bottom edge
            if text_y < 0: text_y = y1 + 20 # Last resort: top of box

        # 4. Draw Text Background (Filled Rectangle) for readability
        # Box coords: (x, y-h-baseline) to (x+w, y+baseline)
        # Note: cv2.putText draws origin at bottom-left of text
        bg_p1 = (text_x, text_y - text_h - 4)
        bg_p2 = (text_x + text_w, text_y + 4)
        
        # Draw background (same color as box, but solid)
        cv2.rectangle(frame, bg_p1, bg_p2, color, -1)
        
        # 5. Draw Text (White or Black depending on brightness of color?)
        # For simplicity, White text usually works on colored backgrounds, unless background is very light.
        # Let's assume darker box colors. If not, Black text is safer.
        # A simple heuristic: compute luminance of BGR color
        b, g, r = color
        luminance = 0.299*r + 0.587*g + 0.114*b
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
        
        cv2.putText(frame, full_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
