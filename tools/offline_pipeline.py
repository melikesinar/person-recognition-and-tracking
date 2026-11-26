"""
The full identity pipeline, driven frame by frame from recorded data.

Shared by tools/replay.py (videos) and tools/eval_wildtrack.py (image sequences
with ground truth). Unlike src/__main__.py there are no threads and nothing is
dropped: every given frame is processed in lock-step for all cameras, and all
timing rules run on a *simulated clock* (the recording's own time), so two runs
with the same settings give identical results regardless of computer speed.
"""
import sys
import time
import logging
import datetime as _dt
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.config import Config  # noqa: E402

CLOCK_ORIGIN = 1_000_000.0  # simulated time of the first frame


class EventCounter(logging.Handler):
    """Counts identity events from the log and remembers them until collected."""
    KEYS = {"ENROLL: New person": "new_identity", "ReID match": "reid_match",
            "Enroll Merge": "enroll_merge", "AUTO-MERGE": "auto_merge",
            "CONSOLIDATE": "consolidate", "SWITCH CONFIRMED": "id_switch",
            "REVOCATION": "revocation", "VISUAL GUARD": "visual_guard",
            "NOT SURE": "not_sure", "Still not sure": "not_sure_timeout",
            "TRANSITION": "transition", "NEW PERSON": "new_lookalike"}

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.counts = {v: 0 for v in self.KEYS.values()}
        self.pending = []

    def emit(self, record):
        msg = record.getMessage()
        for key, name in self.KEYS.items():
            if key in msg:
                self.counts[name] += 1
                self.pending.append((name, msg))


class OfflinePipeline:
    def __init__(self, cameras: List[dict], process_fps: Optional[float] = None):
        """
        Args:
            cameras: list of {"id", "name", "process_fps"} (as from Config.load_cameras()).
                Config.CAMERA_TOPOLOGY / CAMERA_HOMOGRAPHIES must already be set.
            process_fps: overrides each camera's process_fps (frames per second given
                to step()); controls tracker memory length.
        """
        self.clock = {"now": CLOCK_ORIGIN}
        time.time = lambda: self.clock["now"]

        from ultralytics import YOLO
        from utils.logger import get_logger
        from core.camera_node import CameraNode
        from core.reid_extractor import ReidExtractor
        from core.quality_checker import QualityChecker
        from database.vector_index import VectorIndex
        from database.async_writer import NullDBWriter
        from tracking.person_manager import PersonManager
        from tracking.identity_matcher import IdentityMatcher
        from tracking.identity_verifier import IdentityVerifier
        from tracking.topology import CameraTopology
        from tracking.presence import PresenceRegistry
        from tracking.consolidator import IdentityConsolidator
        from utils.torch_utils import configure_cpu_inference
        import tracking.identity_verifier as iv_mod
        import tracking.person_manager as pm_mod

        clock = self.clock

        class SimDatetime(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return _dt.datetime.fromtimestamp(clock["now"], tz)

        # Identity ages (auto-merge window) use datetime.now(): simulated clock too
        iv_mod.datetime = SimDatetime
        pm_mod.datetime = SimDatetime

        self.counter = EventCounter()
        get_logger().addHandler(self.counter)

        device = torch.device("cpu")
        configure_cpu_inference(Config.REID_NUM_THREADS)
        self.detector = YOLO(Config.YOLO_MODEL_PATH)

        self.db: Dict[int, dict] = {}
        index = VectorIndex()
        self.person_manager = PersonManager(self.db, NullDBWriter(), index)
        self.topology = CameraTopology([c["id"] for c in cameras], Config.CAMERA_TOPOLOGY,
                                       Config.SIMULTANEOUS_WINDOW_SECONDS,
                                       Config.CAMERA_HOMOGRAPHIES, Config.SPATIAL_GATE_PX)
        self.presence = PresenceRegistry(self.topology, {c["id"]: c["name"] for c in cameras},
                                         Config.VISIT_GAP_SECONDS)
        self.resources = {
            "person_manager": self.person_manager, "vector_index": index,
            "identity_matcher": IdentityMatcher(self.person_manager, index, self.topology, self.presence),
            "topology": self.topology, "presence": self.presence,
            "reid_extractor": ReidExtractor(device), "quality_checker": QualityChecker(),
            "detection_threshold": multiprocessing.Value("d", Config.DETECTION_SCORE_THRESHOLD),
            "match_threshold": Config.MATCH_SIMILARITY_THRESHOLD, "use_tta": False, "device": device,
        }
        self.verifier = IdentityVerifier()
        self.consolidator = IdentityConsolidator(self.person_manager, self.presence, self.topology)
        self.nodes = [CameraNode(c["id"], c["name"], process_fps or c["process_fps"], self.resources)
                      for c in cameras]

        # Keep each node's tracker output of the last frame, for evaluation tools
        for node in self.nodes:
            original = node._resolve_identities

            def keep(frame, tracker_results, *rest, _node=node, _orig=original):
                _node.last_tracker_results = tracker_results
                return _orig(frame, tracker_results, *rest)

            node._resolve_identities = keep
            node.last_tracker_results = []

    def step(self, t: float, frames: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        """
        Processes one time step: `frames[i]` is camera i's frame at recording time
        `t` seconds (None if that camera has no frame). Returns annotated frames.
        """
        self.clock["now"] = CLOCK_ORIGIN + t
        annotated = []
        for node, frame in zip(self.nodes, frames):
            if frame is None:
                node.last_tracker_results = []
                continue
            dets = self.detector(frame, classes=0, conf=Config.DETECTION_SCORE_THRESHOLD,
                                 imgsz=Config.DETECTION_IMG_SIZE, verbose=False)[0].boxes.data.cpu().numpy()
            annotated.append(node.process_frame(frame, dets))
        self.verifier.check_redundant_identities(self.person_manager, self.nodes)
        self.consolidator.step(self.nodes, self.clock["now"])
        return annotated

    def current_boxes(self, cam_index: int):
        """[(x1, y1, x2, y2, pid or None)] for every tracked person in the last frame."""
        node = self.nodes[cam_index]
        id_map = node.buffer_manager.track_id_to_permanent_id_map
        out = []
        for box, tid, *_ in node.last_tracker_results:
            x1, y1, x2, y2 = (float(v) for v in box)
            out.append((x1, y1, x2, y2, id_map.get(int(tid))))
        return out

    def summary(self, wall_seconds: float, steps: int) -> List[str]:
        lines = [f"processed steps={steps}  wall={wall_seconds:.0f}s",
                 f"identities alive at end: {len(self.db)}   (created: {self.counter.counts['new_identity']})"]
        lines += [f"  {k:16s} {v}" for k, v in self.counter.counts.items()]
        return lines
