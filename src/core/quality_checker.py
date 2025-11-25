"""
Quality assessment module for person detection crops.
Evaluates multiple criteria to determine enrollment suitability.
"""
import cv2
import numpy as np
from utils.config import Config


class QualityChecker:
    """
    Performs multi-criteria quality assessment on detected person crops
    to decide if they are suitable for enrollment into the Re-ID database.
    """

    def __init__(self):
        """Initialize quality checker with configuration parameters."""
        self.min_bbox_area = Config.MIN_BBOX_AREA_ENROLL
        self.min_sharpness = Config.MIN_SHARPNESS_ENROLL

        # Quality scoring weights
        self.weights = Config.Q_WEIGHTS
        self.min_quality_threshold = Config.Q_MIN_QUALITY_THRESHOLD

    def calculate_sharpness(self, image):
        """
        Calculates image sharpness using Laplacian variance.

        Args:
            image (np.ndarray): BGR image crop

        Returns:
            float: Laplacian variance (higher = sharper)
        """
        if image is None or image.size == 0:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

    def check_aspect_ratio(self, height, width):
        """
        Validates aspect ratio (person should be ~2:1 height/width).

        Args:
            height (int): Crop height in pixels
            width (int): Crop width in pixels

        Returns:
            tuple: (score: float, rejection_reason: str or None)
        """
        if width == 0:
            return 0.0, "zero_width"

        aspect_ratio = height / width

        if aspect_ratio < Config.Q_MIN_ASPECT_RATIO:
            return 0.0, "aspect_too_wide"
        elif aspect_ratio > Config.Q_MAX_ASPECT_RATIO:
            return 0.0, "aspect_too_tall"
        elif Config.Q_ASPECT_RATIO_OPTIMAL_MIN <= aspect_ratio <= Config.Q_ASPECT_RATIO_OPTIMAL_MAX:
            return 1.0, None  # Optimal range
        else:
            return 0.5, None  # Acceptable but not ideal

    def check_bbox_area(self, height, width):
        """
        Validates bounding box area (sufficient detail).

        Args:
            height (int): Crop height in pixels
            width (int): Crop width in pixels

        Returns:
            tuple: (score: float, rejection_reason: str or None)
        """
        area = height * width

        if area < self.min_bbox_area:
            return 0.0, "too_small"
        else:
            # Normalized score (10000 = min, 50000 = excellent)
            score = min(1.0, area / Config.Q_BBOX_AREA_EXCELLENT)
            return score, None

    def check_sharpness(self, crop):
        """
        Validates image sharpness (blur detection).

        Args:
            crop (np.ndarray): BGR image crop

        Returns:
            tuple: (score: float, rejection_reason: str or None)
        """
        sharpness = self.calculate_sharpness(crop)

        if sharpness < self.min_sharpness:
            return 0.0, "too_blurry"
        else:
            # Normalized score (50 = min, 200 = excellent)
            score = min(1.0, sharpness / Config.Q_SHARPNESS_EXCELLENT)
            return score, None

    def check_edge_density(self, crop):
        """
        Evaluates edge density (occlusion and clutter detection).

        Args:
            crop (np.ndarray): BGR image crop

        Returns:
            tuple: (score: float, rejection_reason: str or None)
        """
        try:
            h, w = crop.shape[:2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)

            if edge_density < Config.Q_EDGE_DENSITY_MIN:  # Too smooth = occluded
                return 0.0, "occluded"
            elif edge_density > Config.Q_EDGE_DENSITY_MAX:  # Too busy = cluttered
                return 0.5, None
            else:
                # Optimal range: 0.05 - 0.20
                score = 1.0 if Config.Q_EDGE_DENSITY_OPTIMAL_MIN <= edge_density <= Config.Q_EDGE_DENSITY_OPTIMAL_MAX else 0.7
                return score, None
        except Exception:
            return 0.5, None  # Neutral score on error

    def check_brightness(self, crop):
        """
        Evaluates brightness variance (illumination check).

        Args:
            crop (np.ndarray): BGR image crop

        Returns:
            tuple: (score: float, rejection_reason: str or None)
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            brightness_std = gray.std()
            brightness_mean = gray.mean()

            if brightness_std < Config.Q_BRIGHTNESS_STD_MIN:  # Too uniform
                return 0.0, "poor_illumination"
            elif brightness_mean < Config.Q_BRIGHTNESS_MEAN_MIN or brightness_mean > Config.Q_BRIGHTNESS_MEAN_MAX:  # Too dark/bright
                return 0.3, "extreme_lighting"
            else:
                return 1.0, None
        except Exception:
            return 0.5, None

    def check_position(self, bbox, frame_shape):
        """
        Checks if detection is too close to frame edges (truncation risk).

        Args:
            bbox (list): Bounding box [x1, y1, x2, y2]
            frame_shape (tuple): Frame dimensions (H, W, C)

        Returns:
            tuple: (score: float, rejection_reason: str or None)
        """
        x1, y1, x2, y2 = map(int, bbox)
        frame_h, frame_w = frame_shape[:2]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Distance from edges (normalized)
        margin_x = min(center_x, frame_w - center_x) / frame_w
        margin_y = min(center_y, frame_h - center_y) / frame_h

        if margin_x < Config.Q_FRAME_EDGE_MARGIN or margin_y < Config.Q_FRAME_EDGE_MARGIN:  # Within 5% of edge
            return 0.0, "at_frame_edge"
        else:
            return 1.0, None

    def assess_quality(self, crop, bbox, frame_shape):
        """
        Performs comprehensive quality assessment.

        Args:
            crop (np.ndarray): BGR image crop of the person
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            frame_shape (tuple): Original frame shape (H, W, C)

        Returns:
            tuple: (is_good_quality: bool, reason: str, quality_score: float)
        """
        h, w = crop.shape[:2]

        quality_scores = {}
        rejection_reasons = []

        # 1. Aspect Ratio
        score, reason = self.check_aspect_ratio(h, w)
        quality_scores['aspect'] = score
        if reason:
            rejection_reasons.append(reason)

        # 2. Bounding Box Area
        score, reason = self.check_bbox_area(h, w)
        quality_scores['size'] = score
        if reason:
            rejection_reasons.append(reason)

        # 3. Sharpness
        score, reason = self.check_sharpness(crop)
        quality_scores['sharpness'] = score
        if reason:
            rejection_reasons.append(reason)

        # 4. Edge Density
        score, reason = self.check_edge_density(crop)
        quality_scores['edges'] = score
        if reason:
            rejection_reasons.append(reason)

        # 5. Brightness
        score, reason = self.check_brightness(crop)
        quality_scores['brightness'] = score
        if reason:
            rejection_reasons.append(reason)

        # 6. Position
        score, reason = self.check_position(bbox, frame_shape)
        quality_scores['position'] = score
        if reason:
            rejection_reasons.append(reason)

        # Aggregate Score (weighted average)
        total_score = sum(quality_scores[k] * self.weights[k] for k in quality_scores.keys())
        max_score = sum(self.weights.values())
        normalized_score = total_score / max_score

        # Decision: Require minimum quality AND no hard rejections
        is_good_quality = (normalized_score >= self.min_quality_threshold) and (len(rejection_reasons) == 0)

        # Format reason
        reason_str = ", ".join(rejection_reasons) if rejection_reasons else "good"

        return is_good_quality, reason_str, normalized_score