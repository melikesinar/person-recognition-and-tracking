"""
Label-free identity consistency check for two OVERLAPPING cameras.

When two cameras see the same room, a person standing on one floor spot is seen
by both at the same moment, and must carry the SAME P number in both. The floor
homography (`homographies: "0-1"` in the cameras YAML) maps a foot point from
camera 0 to camera 1, so this can be checked without any ground-truth labels.

Runs tools/replay.py (deterministic) and, on every processed frame, maps each
labelled person's feet from camera 0 into camera 1:
    agree     : the person found there has the same P number
    disagree  : different P number -> one of the two identities is wrong
"strict" only counts unambiguous spots (match within 20 px, nobody else within
50 px), "loose" counts any match within 30 px.

Also reports:
    teleports : a P label jumping > 25% of the frame width within one processed
                frame in the same camera (label moved to another person)
    labelled  : share of detected people that carry a P number

Usage (from the project root):
    python tools/eval_consistency.py --cameras config/cameras.epfl-room.yaml --seconds 150
    python tools/eval_consistency.py --cameras ... --set SPATIAL_REACQ_ENABLED=True
"""
import sys
import argparse
import collections
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import replay  # noqa: E402
from core import camera_node as cn  # noqa: E402
from utils.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--seconds", type=float, default=150.0)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="Override a Config value for this run (Python literal), repeatable")
    args = ap.parse_args()

    for kv in args.set:
        name, value = kv.split("=", 1)
        if not hasattr(Config, name):
            raise SystemExit(f"Unknown Config setting: {name}")
        setattr(Config, name, eval(value, {}, {}))
        print(f"override {name} = {value}")

    frames = {}           # step -> {camera: {pid: foot}}
    counts = {"labelled": 0, "detections": 0, "width": 0}
    step = {"i": 0}
    original = cn.CameraNode.process_frame

    def recording(self, frame, dets):
        out = original(self, frame, dets)
        bm = self.buffer_manager
        feet = {}
        for pid in bm.track_id_to_permanent_id_map.values():
            b = bm.last_known_box_for_pid.get(pid)
            if b is not None:
                feet[pid] = ((b[0] + b[2]) / 2.0, b[3])
        counts["labelled"] += len(feet)
        counts["detections"] += len(dets)
        counts["width"] = frame.shape[1]
        if self.camera_id == 0:
            step["i"] += 1
        frames.setdefault(step["i"], {})[self.camera_id] = feet
        return out

    cn.CameraNode.process_frame = recording
    sys.argv = ["replay", "--cameras", args.cameras, "--seconds", str(args.seconds)]
    replay.main()

    if "0-1" not in Config.CAMERA_HOMOGRAPHIES:
        raise SystemExit("This check needs a '0-1' homography in the cameras file.")
    H = np.asarray(Config.CAMERA_HOMOGRAPHIES["0-1"], dtype=float)

    for label, dmax, sep in (("loose", 30, 0), ("strict", 20, 50)):
        agree = disagree = 0
        for cams in frames.values():
            a, b = cams.get(0, {}), cams.get(1, {})
            if not a or not b:
                continue
            items = list(b.items())
            bxy = np.array([xy for _, xy in items])
            for pid, (x, y) in a.items():
                p = H @ np.array([x, y, 1.0])
                p = p[:2] / p[2]
                d = np.linalg.norm(bxy - p, axis=1)
                order = np.argsort(d)
                if d[order[0]] >= dmax or (sep and len(order) > 1 and d[order[1]] < sep):
                    continue
                if items[order[0]][0] == pid:
                    agree += 1
                else:
                    disagree += 1
        print(f"cross-camera agreement ({label}): {agree}/{agree + disagree} = "
              f"{100.0 * agree / max(1, agree + disagree):.0f}%")

    teleports = 0
    width = counts["width"] or 1
    for cam in (0, 1):
        prev = {}
        for s in sorted(frames):
            cur = frames[s].get(cam, {})
            teleports += sum(1 for pid, xy in cur.items()
                             if pid in prev and abs(xy[0] - prev[pid][0]) > 0.25 * width)
            prev = cur
    print(f"teleports: {teleports}")
    print(f"labelled: {100.0 * counts['labelled'] / max(1, counts['detections']):.0f}% of detected people")


if __name__ == "__main__":
    main()
