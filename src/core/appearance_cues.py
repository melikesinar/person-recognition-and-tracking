"""
Part-based colour cues: head/hair, upper body (top), lower body (trousers/skirt).

Why separate cues next to the OSNet vector:
    OSNet compresses the whole person into one vector. When half of the body is
    hidden (cut by the frame, or behind someone) that vector changes a lot, and it
    cannot tell *which* part made two people look alike. Per-part colour
    descriptors can be switched off individually when a part is not visible, and
    they give an explicit, explainable check: "the top is red in both cameras".

Descriptor (per part):
    * Pixels with enough saturation and brightness are "chromatic": binned by
      hue (12 bins, circular) x saturation (3 bins).
    * The others are "achromatic" (black / grey / white clothes, where hue is
      meaningless noise): binned by brightness (4 bins).
    * Hue bins are smoothed with their neighbours so a slight colour shift
      between two cameras still overlaps.
    * The histogram is L1-normalised and square-rooted, so the dot product of two
      descriptors is the Bhattacharyya coefficient (1 = identical distribution,
      0 = no overlap).

Body proportions are fixed fractions of a full-body box (no pose model needed,
which keeps this cheap on a CPU). When the box touches the top or bottom of the
frame, the person is cut off; the full-body height is then estimated from the
box width and parts that fall mostly outside the frame are marked unavailable.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.config import Config

PART_NAMES = ("head", "upper", "lower")

# Vertical extent of each part as a fraction of full-body height (top = 0).
PART_RANGES: Dict[str, Tuple[float, float]] = {
    "head": (0.00, 0.17),
    "upper": (0.20, 0.52),
    "lower": (0.55, 0.90),   # stop above the feet/shoes and floor
}
# Horizontal crop (fraction of box width) kept for each part, to skip background
# at the sides of the box (arms are also mostly excluded, which is fine).
PART_X_RANGE: Dict[str, Tuple[float, float]] = {
    "head": (0.30, 0.70),
    "upper": (0.20, 0.80),
    "lower": (0.25, 0.75),
}

HUE_BINS, SAT_BINS, ACHRO_BINS = 12, 3, 4
DESCRIPTOR_SIZE = HUE_BINS * SAT_BINS + ACHRO_BINS
SAT_MIN, VAL_MIN = 50, 45          # below these, a pixel counts as achromatic
FULL_BODY_ASPECT = 2.7             # typical height / width of a standing person's box
EDGE_MARGIN_PX = 3                 # box this close to the frame border = cut off
MIN_PART_PIXELS = 60               # smaller regions are too noisy to describe
MIN_VISIBLE_FRACTION = 0.7         # a part must be at least this much inside the box


@dataclass
class PartCues:
    """Colour descriptors for one observation; a part is None when not visible."""
    parts: Dict[str, Optional[np.ndarray]]

    def available(self) -> Tuple[str, ...]:
        return tuple(p for p in PART_NAMES if self.parts.get(p) is not None)


def _describe(region_bgr: np.ndarray) -> Optional[np.ndarray]:
    if region_bgr.size == 0 or region_bgr.shape[0] * region_bgr.shape[1] < MIN_PART_PIXELS:
        return None
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    chromatic = (s >= SAT_MIN) & (v >= VAL_MIN)

    hist = np.zeros(DESCRIPTOR_SIZE, dtype=np.float32)

    # Chromatic part: hue (OpenCV range 0..179) x saturation
    hb = (h[chromatic] * HUE_BINS) // 180
    sb = np.minimum((s[chromatic] - SAT_MIN) * SAT_BINS // (256 - SAT_MIN), SAT_BINS - 1)
    chroma = np.bincount(hb * SAT_BINS + sb, minlength=HUE_BINS * SAT_BINS).astype(np.float32)
    # Smooth along the (circular) hue axis
    chroma = chroma.reshape(HUE_BINS, SAT_BINS)
    chroma = 0.5 * chroma + 0.25 * np.roll(chroma, 1, axis=0) + 0.25 * np.roll(chroma, -1, axis=0)
    hist[:HUE_BINS * SAT_BINS] = chroma.ravel()

    # Achromatic part: brightness only
    ab = (v[~chromatic] * ACHRO_BINS) // 256
    hist[HUE_BINS * SAT_BINS:] = np.bincount(ab, minlength=ACHRO_BINS).astype(np.float32)

    total = hist.sum()
    if total <= 0:
        return None
    return np.sqrt(hist / total)


def extract_part_cues(crop_bgr: np.ndarray, bbox: Sequence[float],
                      frame_shape: Optional[Tuple[int, ...]] = None) -> PartCues:
    """
    Args:
        crop_bgr: the person's box cut out of the frame.
        bbox: [x1, y1, x2, y2] of that box in frame coordinates.
        frame_shape: frame (h, w, ...) used to detect boxes cut by the frame border.
            None means "assume the box shows the full body" (e.g. dataset crops).
    """
    h, w = crop_bgr.shape[:2]
    parts: Dict[str, Optional[np.ndarray]] = {p: None for p in PART_NAMES}
    if h < 10 or w < 5:
        return PartCues(parts)

    cut_top = cut_bottom = False
    if frame_shape is not None:
        fh = frame_shape[0]
        cut_top = bbox[1] <= EDGE_MARGIN_PX
        cut_bottom = bbox[3] >= fh - EDGE_MARGIN_PX

    # Estimated full-body height and where the visible box sits inside it
    full_h = float(h)
    offset = 0.0  # position of the box top inside the full body, in pixels
    if (cut_top or cut_bottom) and FULL_BODY_ASPECT * w > h:
        full_h = FULL_BODY_ASPECT * w
        if cut_top and not cut_bottom:
            offset = full_h - h        # we see the bottom part of the body
        elif cut_top and cut_bottom:
            return PartCues(parts)     # cut at both ends: proportions unknowable

    for name in PART_NAMES:
        y0f, y1f = PART_RANGES[name]
        y0 = y0f * full_h - offset
        y1 = y1f * full_h - offset
        vis0, vis1 = max(0.0, y0), min(float(h), y1)
        if vis1 <= vis0 or (vis1 - vis0) / (y1 - y0) < MIN_VISIBLE_FRACTION:
            continue
        x0f, x1f = PART_X_RANGE[name]
        region = crop_bgr[int(vis0):int(vis1), int(x0f * w):int(x1f * w)]
        parts[name] = _describe(region)
    return PartCues(parts)


def part_similarities(a: PartCues, b: PartCues) -> Dict[str, float]:
    """Bhattacharyya coefficient per part visible in both observations."""
    out = {}
    for name in PART_NAMES:
        da, db = a.parts.get(name), b.parts.get(name)
        if da is not None and db is not None:
            out[name] = float(np.dot(da, db))
    return out


def fuse_scores(reid_sim: float, part_sims: Dict[str, float]) -> Tuple[float, bool]:
    """
    Combines the OSNet similarity with the per-part colour similarities.

    Each available cue contributes with its configured weight; cues that are not
    visible simply drop out and the remaining weights are renormalised.

    Returns:
        (fused_score, vetoed). vetoed=True when the clothing colours clearly
        contradict the match (both top and bottom very different), regardless of
        how similar OSNet thinks the two people are.
    """
    weights = Config.CUE_WEIGHTS
    total_w = weights["reid"]
    score = weights["reid"] * reid_sim
    for name, sim in part_sims.items():
        total_w += weights[name]
        score += weights[name] * sim
    fused = score / total_w

    clothing = [part_sims[p] for p in ("upper", "lower") if p in part_sims]
    vetoed = len(clothing) == 2 and max(clothing) < Config.COLOR_VETO_THRESHOLD
    return fused, vetoed
