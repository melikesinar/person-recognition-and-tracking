import numpy as np
from typing import List, Tuple, Optional, Dict
from utils.config import Config
from utils.logger import get_logger

logger = get_logger()

class CrossCameraMatcher:
    """
    Handles spatial consistency checks between different camera views.
    Uses Homography matrices to project points from one camera's view to another.
    """

    def __init__(self):
        self.homography_matrices = Config.CAMERA_HOMOGRAPHIES
        self.enabled = Config.SPATIAL_MERGE_ENABLED
        self.spatial_threshold = Config.SPATIAL_MERGE_DISTANCE_THRESHOLD
        
        if self.enabled:
            logger.info(f"Cross-Camera Spatial Matching ENABLED. Loaded {len(self.homography_matrices)} matrices.")
            if not self.homography_matrices:
                logger.warning("No Homography matrices found in Config. Spatial matching will act as pass-through or be inactive.")
        else:
            logger.info("Cross-Camera Spatial Matching DISABLED (Config.SPATIAL_MERGE_ENABLED=False).")

    def get_feet_point(self, bbox: List[float]) -> np.ndarray:
        """
        Extracts the bottom-center point (feet) from a bounding box.
        Args:
            bbox: [x1, y1, x2, y2]
        Returns:
            np.array: [x, y, 1.0] (homogeneous coordinates)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        bottom_y = y2
        return np.array([center_x, bottom_y, 1.0])

    def project_point(self, point: np.ndarray, H: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Projects a point using a homography matrix.
        Args:
            point: [x, y, 1]
            H: 3x3 Homography matrix
        Returns:
            (x, y) projected coordinates or None if invalid
        """
        try:
            # Matrix multiplication: p' = H * p
            projected = np.dot(H, point)
            
            # Normalize homogeneous coordinates (x'/w', y'/w')
            if abs(projected[2]) < 1e-6: # Avoid division by zero
                return None
                
            x = projected[0] / projected[2]
            y = projected[1] / projected[2]
            return (x, y)
        except Exception as e:
            logger.error(f"Projection error: {e}")
            return None

    def calculate_spatial_score(self, 
                              cam_id_a: int, box_a: List[float], 
                              cam_id_b: int, box_b: List[float]) -> float:
        """
        Calculates a spatial consistency score (0.0 to 1.0) between two boxes in different cameras.
        1.0 means perfect spatial alignment (projected point matches target).
        
        Args:
            cam_id_a, box_a: Source camera and box
            cam_id_b, box_b: Target camera and box
            
        Returns:
            float: Score (0.0 to 1.0). Returns 0.0 if no matrix exists.
        """
        if not self.enabled:
            return 0.0

        # Check for direct matrix: A -> B
        key_a_b = f"{cam_id_a}-{cam_id_b}"
        key_b_a = f"{cam_id_b}-{cam_id_a}"
        
        H = None
        source_box, target_box = None, None

        if key_a_b in self.homography_matrices:
            H = np.array(self.homography_matrices[key_a_b])
            source_box, target_box = box_a, box_b
        elif key_b_a in self.homography_matrices:
            H = np.array(self.homography_matrices[key_b_a])
            # Invert the logic: Project B to A
            # Actually, to use H_BA (Matrix for B->A), we project B's point and compare to A
            # So Source is B, Target is A.
            # However, standard Homography is usually invertible. If only A->B is defined, we can use inv(A->B) for B->A.
            # For simplicity, we assume explicit definitions or try to invert.
            # But here, key_b_a exists, so we have explicit B->A matrix.
            source_box, target_box = box_b, box_a
        else:
            # OPTIONAL: Try to invert if only one way is defined? 
            # For now, strict config requirement.
            return 0.0 

        # 1. Get feet point of source
        src_point = self.get_feet_point(source_box)
        
        # 2. Project to target view
        proj_coords = self.project_point(src_point, H)
        if proj_coords is None:
            return 0.0
            
        proj_x, proj_y = proj_coords
        
        # --- NEW: Check if point is inside the target box (Strongest Match) ---
        tx1, ty1, tx2, ty2 = target_box
        box_height = ty2 - ty1
        
        # Check if point is inside the box (with a small margin for error around edges)
        # REFINEMENT: Only accept if in the LOWER 80% of the box.
        # If it hits the head (top 20%), it's likely a person standing IN FRONT of the target point.
        margin = 10.0 
        valid_y_top = ty1 + (box_height * 0.20)
        
        if (tx1 - margin) <= proj_x <= (tx2 + margin) and (valid_y_top - margin) <= proj_y <= (ty2 + margin):
            # Point is inside the bounding box (legs/torso area)!
            # Since we project 'feet', it should ideally be in the lower half, 
            # but being anywhere in the lower body is a very strong spatial signal.
            return 1.0

        # 3. Get feet point of target
        tgt_point = self.get_feet_point(target_box)
        tgt_x, tgt_y = tgt_point[0], tgt_point[1]
        
        # 4. Calculate Euclidean distance
        dist = np.sqrt((proj_x - tgt_x)**2 + (proj_y - tgt_y)**2)
        
        # 5. Normalize to a score
        # If dist <= threshold, score is high. If dist > 2*threshold, score is 0.
        threshold = self.spatial_threshold
        
        if dist > (threshold * 3): # Allow slightly wider margin before dropping to absolute 0
            return 0.0
        
        # Linear decay: 1.0 at dist=0, 0.0 at dist=3*threshold
        score = max(0.0, 1.0 - (dist / (threshold * 3)))
        
        # Boost: If within core threshold, guarantee high score
        if dist <= threshold:
            # Map [0, threshold] to [0.7, 1.0]
            # normalized_d = dist / threshold (0 to 1)
            # score = 1.0 - 0.3 * normalized_d
            score = 1.0 - 0.3 * (dist / threshold)

        return score
