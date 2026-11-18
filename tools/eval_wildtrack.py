"""
Measures identity accuracy on the WILDTRACK dataset (EPFL, 7 overlapping HD
cameras, every person labelled with a fixed id in every view, 2 frames/second).

Download the annotated dataset from
https://www.epfl.ch/labs/cvlab/data/data-wildtrack/ and unzip it; the folder
must contain Image_subsets/C1..C7, annotations_positions/ and calibrations/.

Usage (from the project root):
    python tools/eval_wildtrack.py --data C:/datasets/Wildtrack
    python tools/eval_wildtrack.py --data ... --cams 1,2,3 --frames 100      # quicker
    python tools/eval_wildtrack.py --data ... --set SPATIAL_REACQ_ENABLED=True

Prints IDF1 / IDR / IDP across cameras and within each camera, detection
recall, fragmentation and mixing (see tools/id_metrics.py for definitions).
"""
import sys
import json
import time
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.config import Config  # noqa: E402

# WILDTRACK view number -> calibration file stem
CALIB_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]
FRAME_PERIOD = 0.5  # annotated frames are 2 per second


def load_annotations(root: Path, views, max_frames=None):
    files = sorted((root / "annotations_positions").glob("*.json"))
    if max_frames:
        files = files[:max_frames]
    if not files:
        raise SystemExit(f"No annotation files in {root / 'annotations_positions'}")
    frames, gt = [], {}
    for k, f in enumerate(files):
        frames.append(f.stem)
        for person in json.loads(f.read_text(encoding="utf-8")):
            for v in person.get("views", []):
                if v["viewNum"] not in views or v["xmin"] < 0:
                    continue
                box = (float(v["xmin"]), float(v["ymin"]), float(v["xmax"]), float(v["ymax"]))
                gt.setdefault((v["viewNum"], k), []).append((int(person["personID"]), box))
    return frames, gt


def _read_matrix(path: Path, name: str):
    """Reads a matrix from an OpenCV XML file (either opencv-matrix or plain text)."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    node = fs.getNode(name)
    m = None
    if not node.empty():
        m = node.mat()
        if m is None and node.isString():
            m = np.array(node.string().split(), dtype=float)
    fs.release()
    if m is None:
        el = ET.parse(path).getroot().find(name)
        if el is not None and el.text:
            m = np.array(el.text.split(), dtype=float)
    return None if m is None else np.asarray(m, dtype=float)


def ground_homographies(root: Path, views):
    """
    Image-to-image floor homographies between the selected views, from the
    dataset's camera calibration (floor plane z = 0: H = K [r1 r2 t]). Lens
    distortion is ignored, so they are approximate near image borders.
    Returns {} if the calibration cannot be read.
    """
    H = {}
    try:
        for v in views:
            name = CALIB_NAMES[v]
            K = _read_matrix(root / "calibrations" / "intrinsic_zero" / f"intr_{name}.xml", "camera_matrix")
            rvec = _read_matrix(root / "calibrations" / "extrinsic" / f"extr_{name}.xml", "rvec")
            tvec = _read_matrix(root / "calibrations" / "extrinsic" / f"extr_{name}.xml", "tvec")
            R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            H[v] = K.reshape(3, 3) @ np.column_stack([R[:, 0], R[:, 1], tvec.reshape(3)])
    except Exception as e:  # noqa: BLE001
        print(f"note: calibration not usable ({e}); same-spot rule disabled")
        return {}
    out = {}
    for a in views:
        for b in views:
            if a < b:
                out[f"{a}-{b}"] = (H[b] @ np.linalg.inv(H[a])).tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="WILDTRACK root folder")
    ap.add_argument("--cams", default="1,2,3,4,5,6,7", help="Cameras to use, 1..7")
    ap.add_argument("--frames", type=int, default=0, help="Use only the first N annotated frames (0 = all 400)")
    ap.add_argument("--imgsz", type=int, default=960, help="Detector input size (HD cameras need > 416)")
    ap.add_argument("--gate", type=float, default=150.0, help="Same-spot tolerance in pixels (HD)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="Override a Config value for this run (Python literal), repeatable")
    ap.add_argument("--save", help="Write per-box results (JSON lines) here for later analysis")
    args = ap.parse_args()

    root = Path(args.data)
    views = [int(c) - 1 for c in args.cams.split(",")]
    frames, gt = load_annotations(root, set(views), args.frames or None)
    print(f"WILDTRACK: {len(frames)} frames x {len(views)} cameras, "
          f"{sum(len(v) for v in gt.values())} labelled boxes, "
          f"{len({p for v in gt.values() for p, _ in v})} people")

    Config.DETECTION_IMG_SIZE = args.imgsz
    Config.SPATIAL_GATE_PX = args.gate
    Config.CAMERA_HOMOGRAPHIES = ground_homographies(root, views)
    # All WILDTRACK cameras look at the same square
    Config.CAMERA_TOPOLOGY = {"overlapping": [[a, b] for a in views for b in views if a < b]}
    for kv in args.set:
        name, value = kv.split("=", 1)
        if not hasattr(Config, name):
            raise SystemExit(f"Unknown Config setting: {name}")
        setattr(Config, name, eval(value, {}, {}))
        print(f"override {name} = {value}")

    from offline_pipeline import OfflinePipeline
    from id_metrics import evaluate, format_report

    cams = [{"id": v, "name": f"C{v + 1}", "process_fps": 1.0 / FRAME_PERIOD} for v in views]
    wall = time.perf_counter()
    pipe = OfflinePipeline(cams)

    pred = {}
    save = open(args.save, "w", encoding="utf-8") if args.save else None
    for k, stem in enumerate(frames):
        images = []
        for v in views:
            img = cv2.imread(str(root / "Image_subsets" / f"C{v + 1}" / f"{stem}.png"))
            if img is None:
                img = cv2.imread(str(root / "Image_subsets" / f"C{v + 1}" / f"{stem}.jpg"))
            images.append(img)
        pipe.step(k * FRAME_PERIOD, images)
        pipe.counter.pending.clear()
        for i, v in enumerate(views):
            boxes = pipe.current_boxes(i)
            pred[(v, k)] = [(pid, (x1, y1, x2, y2)) for x1, y1, x2, y2, pid in boxes]
            if save:
                for x1, y1, x2, y2, pid in boxes:
                    save.write(json.dumps({"cam": v, "frame": k, "pid": pid,
                                           "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]}) + "\n")
        if (k + 1) % 25 == 0:
            print(f"  frame {k + 1}/{len(frames)}  ({time.perf_counter() - wall:.0f}s)", flush=True)
    if save:
        save.close()

    print()
    for line in pipe.summary(time.perf_counter() - wall, len(frames)):
        print(line)
    print()
    for line in format_report(evaluate(gt, pred)):
        print(line)
    print()
    for line in consolidation_diagnostics(pipe, gt, pred):
        print(line)


def consolidation_diagnostics(pipe, gt, pred, min_purity: float = 0.7):
    """
    For tuning CONSOLIDATE_MIN_SCORE with ground truth: every pair of surviving
    identities is labelled "same person" / "different people" using the ground
    truth (an identity belongs to the true person it was mostly given to, if at
    least `min_purity` of its boxes were that person), then the consolidation
    score of the pair is computed. Shows, per threshold, how many correct merges
    (same) and wrong merges (different) a merge rule "score >= threshold" would make.
    Pairs already proven different are excluded (they are never merged anyway).
    """
    from collections import Counter, defaultdict
    from id_metrics import match_detections

    pairs, _, _ = match_detections(gt, pred)
    per_pid = defaultdict(Counter)
    for _, gid, pid in pairs:
        if pid is not None:
            per_pid[pid][gid] += 1
    owner = {}
    for pid, c in per_pid.items():
        gid, n = c.most_common(1)[0]
        if n >= min_purity * sum(c.values()):
            owner[pid] = gid

    cons = pipe.consolidator
    ids = [p for p in owner if p in pipe.db and len(pipe.db[p].get("vectors", [])) >= 1]
    gal = {p: cons._galleries(p) for p in ids}
    same, diff = [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if pipe.presence.proven_different(a, b):
                continue
            s = cons.pair_score(gal[a], gal[b])[0]
            (same if owner[a] == owner[b] else diff).append(s)

    lines = [f"consolidation diagnostics: {len(ids)} identities with a clear owner, "
             f"{len(same)} same-person pairs and {len(diff)} different-people pairs left to decide"]
    for th in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        lines.append(f"  threshold {th:.2f}: correct merges {sum(s >= th for s in same):4d}   "
                     f"wrong merges {sum(s >= th for s in diff):4d}")
    return lines


if __name__ == "__main__":
    main()
