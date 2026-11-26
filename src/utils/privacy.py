"""
Privacy helpers (Faz 4).

1. Head blurring: the head region of every person box is pixelated on everything
   the system shows or saves (live window, event snapshots, dashboard background
   images). No face detector is needed: the head is the top of a person box, which
   also works when the person faces away. Pixelation (not a light blur) so the
   face cannot be recovered by sharpening.
2. Retention: analytics sessions older than RETENTION_DAYS are deleted at start-up.

The identity data itself is anonymous already: numbers (P1, P2...), appearance
vectors and colour histograms, no names and no stored face images.
"""
import shutil
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from utils.config import Config
from utils.logger import get_logger

logger = get_logger()


def head_region(box, frame_shape) -> tuple:
    """(x1, y1, x2, y2) of the head part of a person box, clipped to the frame."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
    h, w = frame_shape[:2]
    bw, bh = x2 - x1, y2 - y1
    # Full-body box: the head is ~20% of the height. A box showing only the upper
    # body (person close to the camera) is wide, so the head is taller than 20%:
    # 0.6 x width covers both cases (equal for a typical 2.7:1 standing person).
    hh = min(bh, int(max(Config.PRIVACY_HEAD_FRACTION * bh, 0.6 * bw)))
    pad = int(0.1 * bw)
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y1 + hh)


def pixelate_heads(frame: np.ndarray, boxes: Iterable) -> np.ndarray:
    """Pixelates the head region of every box, in place. Returns the frame."""
    for box in boxes:
        x1, y1, x2, y2 = head_region(box, frame.shape)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        blocks = max(2, min(8, (x2 - x1) // 4))
        small = cv2.resize(roi, (blocks, max(2, blocks * (y2 - y1) // max(1, x2 - x1))),
                           interpolation=cv2.INTER_AREA)
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return frame


def apply_retention(sessions_dir: Path, days: float) -> int:
    """Deletes analytics session folders older than `days` (0 = keep forever). Returns how many."""
    if not days or days <= 0 or not sessions_dir.is_dir():
        return 0
    limit = time.time() - days * 86400
    removed = 0
    for d in sessions_dir.iterdir():
        if d.is_dir() and (d / "meta.json").is_file() and d.stat().st_mtime < limit:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    if removed:
        logger.info(f"Retention: deleted {removed} analytics session(s) older than {days:g} days")
    return removed
