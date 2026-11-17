# reid_extractor.py
"""
Appearance (ReID) feature extraction.

One instance of this class is shared by all cameras, so the ReID network is
loaded exactly once (previously: once per camera plus one extra copy).
"""
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from boxmot.appearance.reid.auto_backend import ReidAutoBackend

from utils.config import Config
from utils.logger import get_logger
from utils.torch_utils import configure_cpu_inference

logger = get_logger()


class ReidExtractor:
    """Loads the ReID model (OSNet) once and turns person crops into L2-normalised vectors."""

    def __init__(self, device: torch.device, weights: str = Config.REID_MODEL_PATH,
                 use_clahe: bool = Config.REID_USE_CLAHE):
        self.device = device
        self.use_clahe = use_clahe

        if device.type == "cpu":
            configure_cpu_inference(Config.REID_NUM_THREADS)

        backend = ReidAutoBackend(weights=Path(weights), device=device, half=False).model
        self.model = getattr(backend, "model", None)
        if self.model is None or not callable(self.model):
            raise RuntimeError(f"Re-ID model failed to load from {weights}")
        self.model.eval()
        if device.type == "cpu" and Config.REID_JIT_OPTIMIZE:
            self.model = self._optimize_for_cpu(self.model)

        self.IMAGENET_MEAN = torch.as_tensor(Config.IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.IMAGENET_STD = torch.as_tensor(Config.IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

        # Runtime settings (toggled from the main app)
        self.USE_TTA = Config.USE_TTA_DEFAULT
        self.TTA_HORIZONTAL_FLIP = Config.TTA_HORIZONTAL_FLIP

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.dim = Config.VECTOR_DIMENSION
        logger.info(f"ReidExtractor ready ({Path(weights).name}, device={device}, clahe={use_clahe}).")

    @staticmethod
    def _optimize_for_cpu(model: torch.nn.Module) -> torch.nn.Module:
        """
        Traces and freezes the network with TorchScript. Freezing folds batch-norm into
        the convolutions and removes Python per-layer overhead, which dominates for a
        network made of many small layers like OSNet. Measured on the dev laptop:
        191 -> 95 ms per single crop, 87 -> 50 ms per crop in batches of 4, identical
        output (cosine similarity 1.0). Falls back to the plain model on any error.
        """
        try:
            example = torch.randn(1, 3, Config.REID_INPUT_SIZE[1], Config.REID_INPUT_SIZE[0])
            with torch.inference_mode():
                traced = torch.jit.trace(model, example)
                optimized = torch.jit.optimize_for_inference(torch.jit.freeze(traced.eval()))
                optimized(example)  # first call triggers the optimisation passes
            logger.info("ReID model optimised with TorchScript (CPU).")
            return optimized
        except Exception as e:
            logger.warning(f"TorchScript optimisation failed ({e}); using the plain model.")
            return model

    def set_tta_status(self, status: bool):
        self.USE_TTA = status

    def _apply_clahe(self, crop_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        BGR crop (OpenCV order) -> RGB image of REID_INPUT_SIZE, ready for the network.

        The network was trained on RGB images with ImageNet normalisation. OpenCV
        delivers BGR, so the channel swap here is required; feeding BGR silently
        degrades matching accuracy.
        """
        img = self._apply_clahe(crop_bgr) if self.use_clahe else crop_bgr
        img = cv2.resize(img, Config.REID_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @torch.inference_mode()
    def extract_embeddings_from_crops(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """
        Args:
            crops: list of BGR images (all must be non-empty).
        Returns:
            np.ndarray of shape (N, VECTOR_DIMENSION), each row L2-normalised.
        """
        if len(crops) == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        images = np.stack([self.preprocess(c) for c in crops], axis=0)
        batch = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - self.IMAGENET_MEAN) / self.IMAGENET_STD

        features = self.model(batch)
        if self.USE_TTA and self.TTA_HORIZONTAL_FLIP:
            features = (features + self.model(torch.flip(batch, dims=[3]))) / 2.0

        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype(np.float32)

