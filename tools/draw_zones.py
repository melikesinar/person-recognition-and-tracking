"""
Draw named floor areas (zones) and counting lines on a camera image (Faz 4).
The result is printed as YAML to paste under that camera in the cameras file.

Usage (from the project root):
    python tools/draw_zones.py --cameras config/cameras.yaml --camera 0
    python tools/draw_zones.py --image some_frame.jpg

In the window:
    left click   add a point
    z            finish the points as a ZONE (polygon, at least 3 points); type its name in the terminal
    l            finish the last 2 points as a COUNTING LINE; type its name in the terminal
    u            undo the last point
    c            clear the current points
    s / q        print the YAML and quit (q without printing)

Zones: draw them on the FLOOR. The system places a person at their feet (bottom
centre of the box), so an area is "where the feet are".
Counting lines: click point A, then point B. The arrow shows the "in" direction:
crossing the line in the arrow's direction counts as entering, the other way as leaving.
"""
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.config import Config  # noqa: E402

WIN = "draw zones (z=zone, l=line, u=undo, s=save, q=quit)"
COLORS = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (200, 0, 200), (255, 255, 0), (0, 0, 255)]


def grab_frame(source, skip: int = 30):
    """A frame from a camera or video (skips the first frames, often dark or blurred)."""
    cap = cv2.VideoCapture(source)
    frame = None
    for _ in range(skip):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    return frame


def draw(img, zones, lines, points):
    out = img.copy()
    for k, (name, poly) in enumerate(zones.items()):
        c = COLORS[k % len(COLORS)]
        pts = np.array(poly, np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], c)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        cv2.polylines(out, [pts], True, c, 2)
        cv2.putText(out, name, tuple(pts.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    for name, (a, b) in lines.items():
        a, b = np.array(a, float), np.array(b, float)
        cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), (0, 0, 255), 2)
        mid = (a + b) / 2
        d = b - a
        normal = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1)  # points to the "in" side
        tip = mid + 40 * normal
        cv2.arrowedLine(out, tuple(mid.astype(int)), tuple(tip.astype(int)), (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(out, f"{name} (in)", tuple(tip.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for p in points:
        cv2.circle(out, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1)
    if len(points) > 1:
        cv2.polylines(out, [np.array(points, np.int32)], False, (255, 255, 255), 1)
    return out


def to_yaml(zones, lines) -> str:
    out = []
    if zones:
        out.append("    zones:")
        for name, poly in zones.items():
            out.append(f"      {name}: [{', '.join(f'[{x:.0f}, {y:.0f}]' for x, y in poly)}]")
    if lines:
        out.append("    lines:")
        for name, (a, b) in lines.items():
            out.append(f"      {name}: [[{a[0]:.0f}, {a[1]:.0f}], [{b[0]:.0f}, {b[1]:.0f}]]")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", help="Cameras YAML file")
    ap.add_argument("--camera", type=int, default=0, help="Camera id in that file")
    ap.add_argument("--image", help="Use this image instead of grabbing from the camera")
    args = ap.parse_args()

    zones, lines = {}, {}
    if args.image:
        img = cv2.imread(args.image)
    else:
        if args.cameras:
            Config.CAMERAS_CONFIG_PATH = Path(args.cameras).resolve()
        cam = next((c for c in Config.load_cameras() if c["id"] == args.camera), None)
        if cam is None:
            raise SystemExit(f"Camera {args.camera} not found")
        img = grab_frame(cam["source"])
        zones, lines = dict(cam["zones"]), {k: v for k, v in cam["lines"].items()}
    if img is None:
        raise SystemExit("Could not get an image")
    print(f"image size: {img.shape[1]} x {img.shape[0]} (coordinates are in this image's pixels)")

    points = []
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, lambda e, x, y, *_: points.append([x, y]) if e == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        cv2.imshow(WIN, draw(img, zones, lines, points))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == ord("c"):
            points.clear()
        elif key == ord("z"):
            if len(points) < 3:
                print("a zone needs at least 3 points")
                continue
            name = input("zone name (no spaces, e.g. kasa): ").strip() or f"zone{len(zones) + 1}"
            zones[name] = [list(p) for p in points]
            points.clear()
        elif key == ord("l"):
            if len(points) < 2:
                print("a line needs 2 points")
                continue
            name = input("line name (e.g. giris): ").strip() or f"line{len(lines) + 1}"
            lines[name] = [list(points[-2]), list(points[-1])]
            points.clear()
        elif key in (ord("s"), ord("q")):
            break
    cv2.destroyAllWindows()
    if key == ord("s"):
        print("\n# Paste under this camera in the cameras file (same indentation as 'source:'):")
        print(to_yaml(zones, lines))


if __name__ == "__main__":
    main()
