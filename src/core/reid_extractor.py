# reid_extractor.py
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO  # BoostTrack dependency for internal structure
from boxmot import BoostTrack  # Used to load the underlying Re-ID model
from utils.config import Config


class ReidExtractor:
    """
    Handles loading the Re-ID model (OSNet) and performing batch feature
    extraction with advanced preprocessing and Test-Time Augmentation (TTA).
    """

    def __init__(self, device, tracker_instance):
        """
        Initializes the extractor with the Re-ID model from the tracker.

        Args:
            device (torch.device): CUDA or CPU device.
            tracker_instance (BoostTrack): The BoostTrack instance containing the loaded Re-ID model.
        """
        self.device = device
        self.tracker = tracker_instance

        # Load ImageNet normalization tensors to the correct device
        self.IMAGENET_MEAN = torch.tensor(Config.IMAGENET_MEAN).view(1, 3, 1, 1).to(self.device)
        self.IMAGENET_STD = torch.tensor(Config.IMAGENET_STD).view(1, 3, 1, 1).to(self.device)

        # Runtime settings (to be set externally by the main app)
        self.USE_TTA = Config.USE_TTA_DEFAULT
        self.TTA_HORIZONTAL_FLIP = Config.TTA_HORIZONTAL_FLIP

        # Check if the Re-ID model is loaded correctly
        self.reid_model = getattr(self.tracker, "reid_model", None)
        if self.reid_model is None or not hasattr(self.reid_model, 'model') or not callable(self.reid_model.model):
            raise RuntimeError("Re-ID model failed to load in BoostTrack instance!")
        else:
            print("ReidExtractor initialized successfully with Re-ID model.")

    def set_tta_status(self, status):
        """Updates the Test-Time Augmentation (TTA) flag."""
        self.USE_TTA = status

    def _l2_normalize(self, v):
        """L2 normalization helper (copied from PersonManager for self-containment)."""
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    def _preprocess_crop_advanced(self, crop):
        """
        Applies advanced image preprocessing techniques (e.g., CLAHE).
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to the L (Lightness) channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    @torch.inference_mode()
    def extract_embeddings_from_crops(self, crops):
        """
        Extracts feature embeddings (vectors) for a batch of cropped person images.
        """
        if self.reid_model is None:
            return [self._l2_normalize(np.random.randn(Config.VECTOR_DIMENSION).astype(np.float32)) for _ in crops]

        # Filter and process valid crops
        valid_crops = [c for c in crops if c.size > 0]
        if not valid_crops:
            return []

        # Resize and apply advanced preprocessing
        resized_crops = [
            cv2.resize(
                self._preprocess_crop_advanced(c),
                Config.REID_INPUT_SIZE,
                interpolation=cv2.INTER_LINEAR
            )
            for c in valid_crops
        ]

        # Convert to a batch tensor: [N, H, W, C] -> [N, C, H, W]
        images_np = np.stack(resized_crops, axis=0)
        tensor_batch = torch.from_numpy(images_np).permute(0, 3, 1, 2).float().to(self.device)

        # Normalize to [0, 1] and then apply ImageNet normalization
        tensor_batch = tensor_batch / 255.0

        tensor_batch = (tensor_batch - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # Extract features with optional TTA (Test-Time Augmentation)
        if self.USE_TTA and self.TTA_HORIZONTAL_FLIP:
            # Average features from the original image and its horizontal flip
            features_original = self.reid_model.model(tensor_batch)
            features_flipped = self.reid_model.model(torch.flip(tensor_batch, dims=[3]))
            features = (features_original + features_flipped) / 2.0
        else:
            # Standard feature extraction
            features = self.reid_model.model(tensor_batch)

        # L2 Normalize and convert to numpy array
        embeddings = [self._l2_normalize(v.cpu().numpy()) for v in features]

        return embeddings