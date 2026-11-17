# Multi-camera person tracking & re-identification — entry point
import sys
import math
import time
import queue
import argparse
import threading
import multiprocessing
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from utils.config import Config
from utils.logger import get_logger
from core.camera_node import CameraNode
from core.camera_stream import CameraStream
from core.detector import DetectorWorker
from core.reid_extractor import ReidExtractor
from core.quality_checker import QualityChecker
from database.vector_index import VectorIndex
from database.async_writer import AsyncDBWriter, NullDBWriter
from tracking.person_manager import PersonManager
from tracking.identity_verifier import IdentityVerifier
from tracking.identity_matcher import IdentityMatcher
from tracking.topology import CameraTopology
from tracking.presence import PresenceRegistry

logger = get_logger()

WINDOW_NAME = "Multi-Camera Tracking"


class MultiCameraSystem:
    """
    Main orchestration class.

    Data flow per camera:
        CameraStream (own thread, keeps newest frame)
          -> throttled to the camera's process_fps
          -> DetectorWorker (separate process, YOLO, shared by all cameras)
          -> CameraNode (tracking + identity) using the shared ReidExtractor,
             PersonManager and VectorIndex, so an identity created on one camera
             is searchable from every other camera immediately.
    """

    def __init__(self, cameras: List[dict], headless: bool = False):
        logger.info(f"System initializing with {len(cameras)} camera(s)...")
        self.stop_event = threading.Event()
        self.paused = False
        self.headless = headless

        # --- Global Shared State ---
        self.match_threshold = Config.MATCH_SIMILARITY_THRESHOLD
        self.detection_threshold_value = multiprocessing.Value('d', Config.DETECTION_SCORE_THRESHOLD)
        self.use_tta = Config.USE_TTA_DEFAULT
        self.debug_spatial = False

        # --- Storage (optional MongoDB) ---
        self.mongo_handler = self._connect_database()
        if self.mongo_handler is not None:
            self.async_writer = AsyncDBWriter(self.mongo_handler.persons_collection, self.stop_event)
        else:
            self.async_writer = NullDBWriter()

        self.vector_index = VectorIndex()
        self.known_persons_db: Dict[int, dict] = {}
        self.person_manager = PersonManager(self.known_persons_db, self.async_writer, self.vector_index)
        # Faz 2: where cameras are relative to each other, and where everyone has been
        self.topology = CameraTopology([c["id"] for c in cameras], Config.CAMERA_TOPOLOGY,
                                       Config.SIMULTANEOUS_WINDOW_SECONDS,
                                       Config.CAMERA_HOMOGRAPHIES, Config.SPATIAL_GATE_PX)
        self.presence = PresenceRegistry(self.topology, {c["id"]: c["name"] for c in cameras},
                                         Config.VISIT_GAP_SECONDS)
        logger.info(f"Camera topology: {self.topology.describe()}")
        self.identity_matcher = IdentityMatcher(self.person_manager, self.vector_index,
                                                self.topology, self.presence)
        self.quality_checker = QualityChecker()
        self.global_verifier = IdentityVerifier()

        # --- AI Models ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = 0 if self.device.type == 'cuda' else 'cpu'
        logger.info(f"Using device: {self.device}")

        # Loaded ONCE and shared by all cameras (previously: one copy per camera + one extra)
        self.reid_extractor = ReidExtractor(self.device)

        # Shared detector process
        n = len(cameras)
        self.frame_queue = multiprocessing.Queue(maxsize=n * 2)
        self.detection_queue = multiprocessing.Queue(maxsize=n * 4)
        self.detector_ready = multiprocessing.Event()
        self.detector_worker = DetectorWorker(
            self.frame_queue, self.detection_queue,
            self.detection_threshold_value, device_str, self.detector_ready
        )

        self.global_resources = {
            'person_manager': self.person_manager,
            'vector_index': self.vector_index,
            'identity_matcher': self.identity_matcher,
            'topology': self.topology,
            'presence': self.presence,
            'reid_extractor': self.reid_extractor,
            'quality_checker': self.quality_checker,
            'detection_threshold': self.detection_threshold_value,
            'match_threshold': self.match_threshold,
            'use_tta': self.use_tta,
            'device': self.device,
        }

        # --- Cameras ---
        self.camera_nodes: Dict[int, CameraNode] = {}
        self.streams: Dict[int, CameraStream] = {}
        for cam in cameras:
            self.camera_nodes[cam["id"]] = CameraNode(
                cam["id"], cam["name"], cam["process_fps"], self.global_resources
            )
            self.streams[cam["id"]] = CameraStream(cam["id"], cam["source"])

        self.load_db()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _connect_database():
        if not Config.USE_DATABASE:
            logger.warning("MONGO_URI not set: running in memory-only mode (nothing is persisted).")
            return None
        from database.mongo_handler import MongoHandler
        try:
            return MongoHandler()
        except RuntimeError as e:
            logger.warning(f"MongoDB unavailable ({e}). Continuing in memory-only mode.")
            return None

    def load_db(self):
        if self.mongo_handler is None:
            self.person_manager.set_next_id(1)
            return
        persons = self.mongo_handler.load_all_persons()
        self.known_persons_db.update(persons)
        self.vector_index.load(persons)
        self.person_manager.set_next_id(self.mongo_handler.get_next_id())
        logger.info("System ready. DB loaded.")

    # ------------------------------------------------------------------ main loop

    def run(self, max_seconds: Optional[float] = None):
        self.detector_worker.start()
        logger.info("Waiting for Detector...")
        self.detector_ready.wait()
        for stream in self.streams.values():
            stream.start()
        logger.info("Detector Ready. Streams started.")

        # Frames sent to the detector, waiting for their detections: {cam_id: {frame_idx: frame}}
        self._pending_frames: Dict[int, Dict[int, np.ndarray]] = {cid: {} for cid in self.streams}
        self._pending_lock = threading.Lock()
        start_time = time.time()
        pause_overlay = None
        self.frames_processed = {cid: 0 for cid in self.streams}

        # Frames are fed to the detector from a separate thread, so the detector keeps
        # working on the next frame while the main thread runs tracking + ReID.
        feeder = threading.Thread(target=self._feed_detector, name="detector-feeder", daemon=True)
        feeder.start()

        try:
            while True:
                if max_seconds is not None and time.time() - start_time > max_seconds:
                    break

                if not self.paused:
                    processed_any = self._process_detections()

                    if self.debug_spatial:
                        self._draw_spatial_debug()

                    self.global_verifier.check_redundant_identities(
                        self.person_manager, list(self.camera_nodes.values())
                    )

                    if processed_any and not self.headless:
                        combined = self._compose_display()
                        if combined is not None:
                            pause_overlay = combined
                            cv2.imshow(WINDOW_NAME, combined)
                    if not processed_any:
                        time.sleep(0.005)  # nothing to do yet; don't spin the CPU
                    self._log_stats_periodically()
                else:
                    if pause_overlay is not None and not self.headless:
                        display = pause_overlay.copy()
                        cv2.putText(display, "PAUSED", (display.shape[1] // 2 - 100, display.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.imshow(WINDOW_NAME, display)
                    time.sleep(0.1)

                if not self.headless and self.handle_input() == 'q':
                    break

                self.global_resources['match_threshold'] = self.match_threshold
                self.global_resources['use_tta'] = self.use_tta
        finally:
            self.cleanup()

    def _feed_detector(self):
        """
        Background thread: sends each camera's newest frame to the detector, at most
        process_fps times per second per camera. If the detector is saturated the
        queue is full and the frame is simply skipped (a newer one will follow).
        """
        next_submit_time = {cid: 0.0 for cid in self.streams}
        while not self.stop_event.is_set():
            if self.paused:
                time.sleep(0.05)
                continue
            now = time.time()
            for cid, stream in self.streams.items():
                if now < next_submit_time[cid]:
                    continue
                item = stream.read_new()
                if item is None:
                    continue
                frame_idx, _, frame = item
                with self._pending_lock:
                    self._pending_frames[cid][frame_idx] = frame
                try:
                    self.frame_queue.put_nowait((cid, frame_idx, frame))
                except queue.Full:
                    with self._pending_lock:
                        self._pending_frames[cid].pop(frame_idx, None)
                    continue
                next_submit_time[cid] = now + 1.0 / self.camera_nodes[cid].process_fps
            time.sleep(0.003)

    def _process_detections(self) -> bool:
        """Runs tracking + identity for every detection result that has arrived."""
        processed_any = False
        while True:
            try:
                cid, frame_idx, dets = self.detection_queue.get_nowait()
            except queue.Empty:
                break

            with self._pending_lock:
                frames = self._pending_frames[cid]
                frame = frames.pop(frame_idx, None)
                # Frames older than this one will never be processed: drop them
                for old in [k for k in frames if k < frame_idx]:
                    del frames[old]
            if frame is None:
                continue

            node = self.camera_nodes[cid]
            node.last_display_frame = node.process_frame(frame, dets)
            node.last_detections = dets
            self.frames_processed[cid] += 1
            processed_any = True
        return processed_any

    def _log_stats_periodically(self, interval: float = 10.0):
        """Logs the real processing rate per camera, so CPU overload is visible."""
        now = time.time()
        if not hasattr(self, '_stats_t0'):
            self._stats_t0, self._stats_counts = now, dict(self.frames_processed)
            return
        elapsed = now - self._stats_t0
        if elapsed < interval:
            return
        rates = {cid: (self.frames_processed[cid] - self._stats_counts[cid]) / elapsed
                 for cid in self.frames_processed}
        target = {cid: n.process_fps for cid, n in self.camera_nodes.items()}
        logger.info("Processing FPS per camera: " +
                    ", ".join(f"cam{cid}={rates[cid]:.1f}/{target[cid]:g}" for cid in rates) +
                    f" | identities: {len(self.known_persons_db)}")
        self._stats_t0, self._stats_counts = now, dict(self.frames_processed)

    def _compose_display(self) -> Optional[np.ndarray]:
        """Tiles all camera views into a grid (works for any number of cameras)."""
        frames = [n.last_display_frame for n in self.camera_nodes.values() if n.last_display_frame is not None]
        if not frames:
            return None
        tile_h, tile_w = 360, 640
        tiles = [cv2.resize(f, (tile_w, tile_h)) for f in frames]
        cols = math.ceil(math.sqrt(len(tiles)))
        rows = math.ceil(len(tiles) / cols)
        tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
        return np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])

    def _draw_spatial_debug(self):
        """Projects each person's feet into the other cameras using the homography matrices."""
        matcher = self.global_verifier.cross_camera_matcher
        for node_a in self.camera_nodes.values():
            dets_a = getattr(node_a, 'last_detections', None)
            if dets_a is None:
                continue
            for box_a in dets_a:
                feet_a = matcher.get_feet_point(box_a[:4])
                for node_b in self.camera_nodes.values():
                    if node_a.camera_id == node_b.camera_id or node_b.last_display_frame is None:
                        continue
                    key = f"{node_a.camera_id}-{node_b.camera_id}"
                    if key not in matcher.homography_matrices:
                        continue
                    proj = matcher.project_point(feet_a, np.array(matcher.homography_matrices[key]))
                    if proj:
                        px, py = int(proj[0]), int(proj[1])
                        cv2.circle(node_b.last_display_frame, (px, py), 6, (0, 255, 255), -1)
                        cv2.circle(node_b.last_display_frame, (px, py), 10, (0, 0, 0), 2)

    # ------------------------------------------------------------------ input / shutdown

    def handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): return 'q'
        if key == ord('p'):
            self.paused = not self.paused
            print(f"System {'PAUSED' if self.paused else 'RESUMED'}")
        if key == ord('u'): self.detection_threshold_value.value = min(1.0, self.detection_threshold_value.value + 0.05)
        if key == ord('j'): self.detection_threshold_value.value = max(0.0, self.detection_threshold_value.value - 0.05)
        if key == ord('y'): self.match_threshold = min(1.0, self.match_threshold + 0.05)
        if key == ord('h'): self.match_threshold = max(0.0, self.match_threshold - 0.05)
        if key == ord('d'):
            self.debug_spatial = not self.debug_spatial
            print(f"Spatial Debug Mode: {'ON' if self.debug_spatial else 'OFF'}")
        if key == ord('t'):
            self.use_tta = not self.use_tta
            self.reid_extractor.set_tta_status(self.use_tta)
        if key == ord('r'):
            logger.warning("RESET DB TRIGGERED BY USER...")
            self.person_manager.reset_all()
            for node in self.camera_nodes.values():
                node.buffer_manager.reset_all()
        return key

    def cleanup(self):
        logger.info("System shutting down...")
        self.stop_event.set()
        for stream in self.streams.values():
            stream.stop()
        self.detector_worker.stop()
        self.async_writer.shutdown()
        if not self.headless:
            cv2.destroyAllWindows()
        if self.mongo_handler is not None:
            self.mongo_handler.close()
        logger.info(f"Frames processed per camera: {self.frames_processed if hasattr(self, 'frames_processed') else {}}; "
                    f"identities: {len(self.known_persons_db)}")
        self._write_routes()

    def _write_routes(self):
        """Saves every person's route (camera visits and dwell times) as CSV."""
        from datetime import datetime
        path = Config.OUTPUT_DIR / f"routes_{datetime.now():%Y%m%d_%H%M%S}.csv"
        n = self.presence.export_csv(path)
        logger.info(f"Routes: {n} visits written to {path}")
        for line in self.presence.summary_lines():
            logger.info(f"  {line}")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-camera person tracking & re-identification")
    parser.add_argument("--cameras", help="Path to a cameras YAML file (default: config/cameras.yaml)")
    parser.add_argument("--headless", action="store_true", help="Run without a display window")
    parser.add_argument("--max-seconds", type=float, help="Stop automatically after N seconds")
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    if args.cameras:
        from pathlib import Path
        Config.CAMERAS_CONFIG_PATH = Path(args.cameras).resolve()

    Config.validate()
    Config.print_config()

    try:
        app = MultiCameraSystem(Config.load_cameras(), headless=args.headless)
        app.run(max_seconds=args.max_seconds)
    except Exception as e:
        logger.critical(f"CRITICAL SYSTEM FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
