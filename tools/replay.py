"""
Deterministic offline replay of recorded videos through the full pipeline.

Unlike src/__main__.py (real-time, threads, frames dropped when the CPU is busy),
this processes every N-th frame of each video in lock-step, in a single process,
on the video's own clock, so two runs with the same settings give the same
result. Use it to compare configurations fairly and to inspect what happened at
a given moment.

Usage (from the project root):
    python tools/replay.py --cameras config/cameras.epfl-room.yaml --seconds 120
    python tools/replay.py --cameras ... --mode osnet          # Faz 0 decision rule
    python tools/replay.py --cameras ... --snapshots out_dir   # save frames at key events
    python tools/replay.py --cameras ... --routes routes.csv   # every person's camera visits

Modes:
    multicue : OSNet + clothing colours (Faz 1+, default)
    osnet    : OSNet only with the Faz 0 thresholds (0.85 / 0.88), for comparison
"""
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_pipeline import OfflinePipeline, CLOCK_ORIGIN  # noqa: E402
from utils.config import Config  # noqa: E402


def apply_mode(mode: str):
    if mode == "osnet":
        Config.CUE_WEIGHTS = {"reid": 1.0, "upper": 0.0, "lower": 0.0, "head": 0.0}
        Config.COLOR_VETO_THRESHOLD = -1.0
        Config.MATCH_SIMILARITY_THRESHOLD = 0.85
        Config.REENTRY_SIMILARITY_THRESHOLD = 0.88


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
    cams = Config.load_cameras()

    wall_start = time.perf_counter()
    pipe = OfflinePipeline(cams, process_fps=args.fps)
    print(f"topology: {pipe.topology.describe()}")

    caps = [cv2.VideoCapture(c["source"]) for c in cams]
    native = caps[0].get(cv2.CAP_PROP_FPS) or 25.0
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * (cap.get(cv2.CAP_PROP_FPS) or native)))
    fps = args.fps or cams[0]["process_fps"]
    step_every = max(1, int(round(native / fps)))

    snap_dir = Path(args.snapshots) if args.snapshots else None
    if snap_dir:
        snap_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    t = args.start
    for fi in range(int(args.seconds * native)):
        frames = []
        for cap in caps:
            ok, f = cap.read()
            frames.append(f if ok else None)
        if all(f is None for f in frames):
            break
        if fi % step_every:
            continue
        t = args.start + fi / native
        annotated = pipe.step(t, frames)
        processed += 1

        pending = pipe.counter.pending
        if snap_dir and pending and annotated:
            canvas = np.hstack([cv2.resize(a, (480, 384)) for a in annotated])
            kinds = sorted({k for k, _ in pending})
            cv2.imwrite(str(snap_dir / f"t{t:07.1f}s_{'+'.join(kinds)}.jpg"), canvas)
            with open(snap_dir / "events.txt", "a", encoding="utf-8") as fh:
                for _, msg in pending:
                    fh.write(f"{t:7.1f}s  {msg}\n")
        pending.clear()

    print(f"\nmode={args.mode}  video={args.seconds:.0f}s")
    for line in pipe.summary(time.perf_counter() - wall_start, processed):
        print(line)

    pipe.presence.close_stale(CLOCK_ORIGIN + t + Config.VISIT_GAP_SECONDS + 1)
    print("\nroutes (visits >= 1 s):")
    for line in pipe.presence.summary_lines(origin=CLOCK_ORIGIN):
        print("  " + line)
    if args.routes:
        n = pipe.presence.export_csv(Path(args.routes), origin=CLOCK_ORIGIN)
        print(f"{n} visits written to {args.routes}")


if __name__ == "__main__":
    main()
