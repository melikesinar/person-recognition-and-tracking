"""
Session recorder (Faz 4): writes what the system sees to disk, so that the
dashboard (dashboard/app.py) can compute statistics, live or afterwards.

One folder per run:  output/sessions/<YYYYmmdd_HHMMSS>[_label]/
    meta.json            cameras (names, zones, counting lines, image size),
                         clock type, sample period, privacy settings
    positions.csv        t, cam, pid, x, y   - each identified person's foot point,
                         at most once per ANALYTICS_SAMPLE_SECONDS per camera
    merges.csv           t, keep, drop       - identities found to be one person
                         (the dashboard applies these, so history stays correct)
    health.csv           t, cam, fps, target_fps, status
    background_cam<N>.jpg  empty-scene image of each camera for the heat map

Only numbers and coordinates are stored: no person crops. The background image
is the per-pixel median of frames taken a few seconds apart, which removes
walking people, and heads are pixelated in every sample before that.
"""
import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.config import Config
from utils.logger import get_logger
from utils.privacy import pixelate_heads

logger = get_logger()

BG_MAX_WIDTH = 960  # background images are stored at most this wide


class SessionRecorder:
    def __init__(self, cameras: List[dict], label: Optional[str] = None, clock: str = "wall",
                 origin: Optional[float] = None, root: Optional[Path] = None, topology: Optional[dict] = None):
        """
        Args:
            cameras: [{"id", "name", "zones"?, "lines"?}] as from Config.load_cameras().
            label: appended to the folder name (e.g. "epfl", "wildtrack").
            clock: "wall" (live, times are Unix time) or "video" (recorded data,
                times are `origin` + seconds into the recording).
            topology: the cameras' topology spec, stored for reference.
        """
        root = Path(root or Config.OUTPUT_DIR / "sessions")
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{label}" if label else "")
        self.dir = root / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sample_period = float(Config.ANALYTICS_SAMPLE_SECONDS)
        self._lock = threading.Lock()
        self._last_sample: Dict[tuple, float] = {}
        self._last_t = 0.0
        self._last_flush = time.monotonic()
        self._bg: Dict[int, list] = {}
        self._bg_next: Dict[int, float] = {}
        self._frame_size: Dict[int, list] = {}

        self.meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "clock": clock,
            "origin": origin,
            "sample_period": self.sample_period,
            "visit_gap": Config.VISIT_GAP_SECONDS,
            "privacy": {"heads_pixelated": Config.PRIVACY_BLUR_HEADS,
                        "retention_days": Config.RETENTION_DAYS},
            "topology": topology or {},
            "cameras": [{"id": c["id"], "name": c["name"], "zones": c.get("zones") or {},
                         "lines": c.get("lines") or {}} for c in cameras],
        }
        self._write_meta()

        self._files = {}
        self._writers = {}
        for key, header in (("positions", ["t", "cam", "pid", "x", "y"]),
                            ("merges", ["t", "keep", "drop"]),
                            ("health", ["t", "cam", "fps", "target_fps", "status"])):
            f = open(self.dir / f"{key}.csv", "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(header)
            self._files[key], self._writers[key] = f, w
        logger.info(f"Analytics: recording session to {self.dir}")

    # --- Data -------------------------------------------------------------------

    def position(self, pid: int, cam: int, t: float, foot: Optional[tuple]):
        """A person's foot point; kept at most once per sample period per (person, camera)."""
        if foot is None:
            return
        with self._lock:
            self._last_t = max(self._last_t, t)
            key = (pid, cam)
            last = self._last_sample.get(key)
            if last is not None and t - last < self.sample_period - 1e-6:
                return
            self._last_sample[key] = t
            self._writers["positions"].writerow([f"{t:.2f}", cam, pid, f"{foot[0]:.1f}", f"{foot[1]:.1f}"])
            self._maybe_flush()

    def merge(self, keep: int, drop: int):
        with self._lock:
            self._writers["merges"].writerow([f"{self._last_t:.2f}", keep, drop])
            self._maybe_flush(force=True)

    def health(self, t: float, cam: int, fps: float, target_fps: float, status: str):
        with self._lock:
            self._writers["health"].writerow([f"{t:.2f}", cam, f"{fps:.2f}", f"{target_fps:g}", status])
            self._maybe_flush(force=True)

    # --- Background images --------------------------------------------------------

    def wants_frame(self, cam: int, t: float) -> bool:
        """True when a new background sample of this camera is due (copy the raw frame then)."""
        return len(self._bg.get(cam, [])) < Config.ANALYTICS_BG_SAMPLES and t >= self._bg_next.get(cam, -1e18)

    def add_background(self, cam: int, frame: np.ndarray, boxes, t: float):
        """Adds one raw frame (people's heads get pixelated); writes the median image."""
        h, w = frame.shape[:2]
        if cam not in self._frame_size:
            self._frame_size[cam] = [w, h]
            for c in self.meta["cameras"]:
                if c["id"] == cam:
                    c["frame_size"] = [w, h]
            self._write_meta()
        img = pixelate_heads(frame, boxes)
        scale = min(1.0, BG_MAX_WIDTH / w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        samples = self._bg.setdefault(cam, [])
        samples.append(img)
        self._bg_next[cam] = t + Config.ANALYTICS_BG_INTERVAL
        n = len(samples)
        if n in (1, 5) or n % 5 == 0:
            median = np.median(np.stack(samples), axis=0).astype(np.uint8)
            cv2.imwrite(str(self.dir / f"background_cam{cam}.jpg"), median, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if n >= Config.ANALYTICS_BG_SAMPLES:
            self._bg[cam] = [None] * n  # free the images; the length marks this camera as done

    # --- Files ------------------------------------------------------------------

    def _write_meta(self):
        with open(self.dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def _maybe_flush(self, force: bool = False):
        now = time.monotonic()
        if force or now - self._last_flush > 2.0:
            for f in self._files.values():
                f.flush()
            self._last_flush = now

    def close(self):
        with self._lock:
            for f in self._files.values():
                f.close()
        logger.info(f"Analytics: session saved in {self.dir}")
