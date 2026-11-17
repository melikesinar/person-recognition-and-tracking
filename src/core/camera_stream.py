"""
Threaded frame reader for a single camera source (webcam, video file or network stream).

Why a thread per camera:
    cv2.VideoCapture.read() blocks until the next frame arrives. Reading several
    cameras one after another in the main loop means the slowest camera stalls all
    others, and frames pile up in the driver buffer (growing latency). A background
    reader that always keeps only the *latest* frame avoids both problems.
"""
import time
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger()


class CameraStream:
    """Reads frames from one source in a background thread and keeps only the newest one."""

    def __init__(self, camera_id: int, source: Any, loop_video: bool = True):
        self.camera_id = camera_id
        self.source = source
        self.loop_video = loop_video
        self.is_file = isinstance(source, str) and Path(source).is_file()

        self.cap = cv2.VideoCapture(source)
        self.opened = self.cap.isOpened()
        if not self.opened:
            logger.error(f"[Cam {camera_id}] Could not open source: {source}")

        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.opened else 0
        # Some webcams/streams report 0 or absurd values
        self.native_fps = fps if 1.0 <= fps <= 120.0 else 30.0

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_idx = -1
        self._frame_time = 0.0
        self._last_returned_idx = -1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, name=f"cam{camera_id}-reader", daemon=True)

    def start(self) -> "CameraStream":
        if self.opened:
            self._thread.start()
        return self

    def _reader(self):
        frame_interval = 1.0 / self.native_fps
        next_due = time.time()
        idx = 0
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file and self.loop_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning(f"[Cam {self.camera_id}] Stream ended or read failed.")
                time.sleep(0.5)
                continue

            with self._lock:
                self._frame = frame
                self._frame_idx = idx
                self._frame_time = time.time()
            idx += 1

            # Video files would otherwise be decoded as fast as possible.
            # Play them at their recorded speed so they behave like a live camera.
            if self.is_file:
                next_due += frame_interval
                delay = next_due - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_due = time.time()

    def read_new(self) -> Optional[Tuple[int, float, np.ndarray]]:
        """
        Returns (frame_idx, capture_time, frame) if a frame newer than the last
        returned one is available, otherwise None. The frame is not copied;
        the reader thread never mutates a frame after publishing it.
        """
        with self._lock:
            if self._frame is None or self._frame_idx == self._last_returned_idx:
                return None
            self._last_returned_idx = self._frame_idx
            return self._frame_idx, self._frame_time, self._frame

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.cap.release()
