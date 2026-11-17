"""
Deterministic offline replay of recorded videos through the full pipeline.

Unlike src/__main__.py (real-time, threads, frames dropped when the CPU is busy),
this processes every N-th frame of each video in lock-step, in a single process,
so two runs with the same settings give the same result. Use it to compare
configurations fairly and to inspect what happened at a given moment.

Usage (from the project root):
    python tools/replay.py --cameras config/cameras.epfl-room.yaml --seconds 120
    python tools/replay.py --cameras ... --mode osnet          # Faz 0 decision rule
    python tools/replay.py --cameras ... --snapshots out_dir   # save frames at key events

Modes:
    multicue : OSNet + clothing colours (Faz 1, default)
    osnet    : OSNet only with the Faz 0 thresholds (0.85 / 0.88), for comparison
"""
import sys
import time
import logging
import argparse
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.config import Config  # noqa: E402


def apply_mode(mode: str):
    if mode == "osnet":
        Config.CUE_WEIGHTS = {"reid": 1.0, "upper": 0.0, "lower": 0.0, "head": 0.0}
        Config.COLOR_VETO_THRESHOLD = -1.0
        Config.MATCH_SIMILARITY_THRESHOLD = 0.85
        Config.REENTRY_SIMILARITY_THRESHOLD = 0.88


class EventCounter(logging.Handler):
    """Counts identity events from the log and remembers when they happened."""
    KEYS = {"ENROLL: New person": "new_identity", "ReID match": "reid_match",
            "Enroll Merge": "enroll_merge", "Enroll Conflict": "enroll_conflict",
            "AUTO-MERGE": "auto_merge", "SWITCH CONFIRMED": "id_switch",
            "REVOCATION": "revocation", "VISUAL GUARD": "visual_guard",
            "NOT SURE": "not_sure", "Still not sure": "not_sure_timeout",
            "TRANSITION": "transition", "NEW PERSON": "new_lookalike"}

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.counts = {v: 0 for v in self.KEYS.values()}
        self.pending = []  # events since the last frame, for snapshots

    def emit(self, record):
        msg = record.getMessage()
        for key, name in self.KEYS.items():
            if key in msg:
                self.counts[name] += 1
                self.pending.append((name, msg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--mode", choices=["multicue", "osnet"], default="multicue")
    ap.add_argument("--seconds", type=float, default=120.0, help="Video time to process")
    ap.add_argument("--start", type=float, default=0.0, help="Start offset in seconds")
    ap.add_argument("--fps", type=float, default=None, help="Processing FPS (default: from cameras file)")
    ap.add_argument("--snapshots", help="Folder to save annotated frames at identity events")
    ap.add_argument("--routes", help="CSV file to write every person's camera visits to")
    ap.add_argument("--margin", type=float, help="Override AMBIGUITY_MARGIN (0 disables 'not sure')")
    args = ap.parse_args()

    apply_mode(args.mode)
    if args.margin is not None:
        Config.AMBIGUITY_MARGIN = args.margin
    Config.CAMERAS_CONFIG_PATH = Path(args.cameras).resolve()

    # The pipeline's timing rules (update intervals, short-term memory, ID locks...)
    # read time.time(). Replace it with the *video* clock, so results depend only on
    # the video and settings, not on how fast this computer happens to be.
    video_clock = {"now": 1_000_000.0}
    time.time = lambda: video_clock["now"]
    wall_start = time.perf_counter()

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
    from utils.torch_utils import configure_cpu_inference

    # Identity ages (auto-merge window) use datetime.now(); route those to the video clock too
    import datetime as _dt
    import tracking.identity_verifier as _iv
    import tracking.person_manager as _pm

    class VideoDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime.fromtimestamp(video_clock["now"], tz)

    _iv.datetime = VideoDatetime
    _pm.datetime = VideoDatetime

    logger = get_logger()
    counter = EventCounter()
    logger.addHandler(counter)

    cams = Config.load_cameras()
    device = torch.device("cpu")
    configure_cpu_inference(Config.REID_NUM_THREADS)
    detector = YOLO(Config.YOLO_MODEL_PATH)

    db = {}
    index = VectorIndex()
    pm = PersonManager(db, NullDBWriter(), index)
    topology = CameraTopology([c["id"] for c in cams], Config.CAMERA_TOPOLOGY,
                              Config.SIMULTANEOUS_WINDOW_SECONDS,
                              Config.CAMERA_HOMOGRAPHIES, Config.SPATIAL_GATE_PX)
    presence = PresenceRegistry(topology, {c["id"]: c["name"] for c in cams}, Config.VISIT_GAP_SECONDS)
    print(f"topology: {topology.describe()}")
    resources = {
        "person_manager": pm, "vector_index": index,
        "identity_matcher": IdentityMatcher(pm, index, topology, presence),
        "topology": topology, "presence": presence,
        "reid_extractor": ReidExtractor(device), "quality_checker": QualityChecker(),
        "detection_threshold": multiprocessing.Value("d", Config.DETECTION_SCORE_THRESHOLD),
        "match_threshold": Config.MATCH_SIMILARITY_THRESHOLD, "use_tta": False, "device": device,
    }
    verifier = IdentityVerifier()

    nodes, caps, steps = [], [], []
    for cam in cams:
        fps = args.fps or cam["process_fps"]
        cap = cv2.VideoCapture(cam["source"])
        native = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * native))
        nodes.append(CameraNode(cam["id"], cam["name"], fps, resources))
        caps.append(cap)
        steps.append(max(1, int(round(native / fps))))

    snap_dir = Path(args.snapshots) if args.snapshots else None
    if snap_dir:
        snap_dir.mkdir(parents=True, exist_ok=True)

    native0 = caps[0].get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(args.seconds * native0)
    processed = 0
    for fi in range(total_frames):
        frames = []
        for cap in caps:
            ok, f = cap.read()
            frames.append(f if ok else None)
        if all(f is None for f in frames):
            break
        if fi % steps[0]:
            continue
        video_clock["now"] = 1_000_000.0 + args.start + fi / native0
        annotated = []
        for node, f in zip(nodes, frames):
            if f is None:
                continue
            dets = detector(f, classes=0, conf=Config.DETECTION_SCORE_THRESHOLD,
                            imgsz=Config.DETECTION_IMG_SIZE, verbose=False)[0].boxes.data.cpu().numpy()
            annotated.append(node.process_frame(f, dets))
        verifier.check_redundant_identities(pm, nodes)
        processed += 1

        if snap_dir and counter.pending and annotated:
            video_t = args.start + fi / native0
            canvas = np.hstack([cv2.resize(a, (480, 384)) for a in annotated])
            kinds = sorted({k for k, _ in counter.pending})
            name = f"t{video_t:07.1f}s_{'+'.join(kinds)}.jpg"
            cv2.imwrite(str(snap_dir / name), canvas)
            with open(snap_dir / "events.txt", "a", encoding="utf-8") as fh:
                for _, msg in counter.pending:
                    fh.write(f"{video_t:7.1f}s  {msg}\n")
        counter.pending.clear()

    print(f"\nmode={args.mode}  video={args.seconds:.0f}s  processed steps={processed}  "
          f"wall={time.perf_counter() - wall_start:.0f}s")
    print(f"identities alive at end: {len(db)}   (created: {counter.counts['new_identity']})")
    for k, v in counter.counts.items():
        print(f"  {k:16s} {v}")

    origin = 1_000_000.0  # video clock starts here, so route times are video seconds
    presence.close_stale(video_clock["now"] + Config.VISIT_GAP_SECONDS + 1)
    print("\nroutes (visits >= 1 s):")
    for line in presence.summary_lines(origin=origin):
        print("  " + line)
    if args.routes:
        n = presence.export_csv(Path(args.routes), origin=origin)
        print(f"{n} visits written to {args.routes}")


if __name__ == "__main__":
    main()
