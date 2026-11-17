"""
Camera topology: which cameras see the same area, and how long it takes at
minimum to walk from one camera to another (Faz 2).

Appearance alone cannot separate two people who wear the same clothes. Space and
time can: a person cannot be in two places at once, and cannot get from the door
camera to the corridor camera faster than walking allows. These rules are hard
constraints. A candidate identity that would break them is removed before the
appearance decision is made.

Defined in the cameras YAML file (see config/cameras.yaml):

    topology:
      overlapping: [[0, 1]]          # these cameras see (partly) the same area
      default_min_seconds: 0         # minimum travel time between cameras not listed below
      transitions:                   # walking paths between cameras (both directions)
        - {cameras: [0, 2], min_seconds: 3, max_seconds: 60}
        - {cameras: [2, 3], min_seconds: 5, max_seconds: 120}

Rules derived from it:
    * Overlapping cameras: no constraint, a person can be visible in both at once.
    * Any other pair: the person cannot be visible in both at the same time.
    * The minimum travel time between two cameras is the shortest path over
      `transitions` (e.g. 0 -> 2 -> 3 takes at least 3 + 5 = 8 s). Pairs with no
      path use `default_min_seconds`.
    * max_seconds is informational only (used in reports). A person may stop at a
      coffee machine between cameras, so a long gap is never a reason to reject.
    * Overlapping cameras with a floor homography (`homographies:` in the YAML):
      a person visible in both at once must stand on the same floor spot. If P2
      is at the door in Cam 0, someone at the window in Cam 1 cannot be P2.
"""
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger()

INF = float("inf")


class CameraTopology:
    def __init__(self, camera_ids: Iterable[int], spec: Optional[dict] = None,
                 simultaneous_window: float = 1.0, homographies: Optional[dict] = None,
                 spatial_gate_px: float = 60.0):
        """
        Args:
            camera_ids: ids of all cameras in the installation.
            spec: the `topology:` section of the cameras YAML (may be None/empty).
            simultaneous_window: two sightings closer than this (seconds) count as
                "at the same time". Covers small clock/processing differences
                between camera streams.
            homographies: {"a-b": 3x3 matrix} mapping floor points (feet) from
                the image of camera a to the image of camera b (overlapping cameras).
            spatial_gate_px: how far (pixels, in the target camera) a mapped foot
                point may be from the person's feet and still be "the same spot".
                Generous on purpose: homographies are only approximately right.
        """
        spec = spec or {}
        self.camera_ids = sorted(set(int(c) for c in camera_ids))
        self.simultaneous_window = float(simultaneous_window)
        self.default_min = float(spec.get("default_min_seconds", 0.0))

        self._overlap = set()
        for pair in spec.get("overlapping") or []:
            a, b = int(pair[0]), int(pair[1])
            self._overlap.add(frozenset((a, b)))

        self._edges: Dict[frozenset, dict] = {}
        for t in spec.get("transitions") or []:
            a, b = (int(c) for c in t["cameras"])
            self._edges[frozenset((a, b))] = {
                "min": float(t.get("min_seconds", 0.0)),
                "max": float(t["max_seconds"]) if t.get("max_seconds") is not None else None,
            }

        unknown = {c for p in list(self._overlap) + list(self._edges) for c in p} - set(self.camera_ids)
        if unknown:
            raise ValueError(f"topology refers to unknown camera id(s): {sorted(unknown)}")

        self._min_travel = self._shortest_paths()

        self.spatial_gate = float(spatial_gate_px)
        self._H: Dict[Tuple[int, int], np.ndarray] = {}
        for key, m in (homographies or {}).items():
            a, b = (int(x) for x in str(key).split("-"))
            H = np.asarray(m, dtype=float)
            self._H[(a, b)] = H
            self._H.setdefault((b, a), np.linalg.inv(H))

    def _shortest_paths(self) -> Dict[int, Dict[int, float]]:
        """All-pairs minimum travel time over the transition graph (Floyd-Warshall)."""
        ids = self.camera_ids
        d = {a: {b: (0.0 if a == b else INF) for b in ids} for a in ids}
        for pair, e in self._edges.items():
            a, b = tuple(pair)
            d[a][b] = d[b][a] = min(d[a][b], e["min"])
        for k in ids:
            for i in ids:
                for j in ids:
                    if d[i][k] + d[k][j] < d[i][j]:
                        d[i][j] = d[i][k] + d[k][j]
        return d

    # --- Queries ---------------------------------------------------------------

    def overlaps(self, a: int, b: int) -> bool:
        return a == b or frozenset((a, b)) in self._overlap

    def min_travel(self, a: int, b: int) -> float:
        if self.overlaps(a, b):
            return 0.0
        t = self._min_travel.get(a, {}).get(b, INF)
        return self.default_min if t == INF else t

    def max_travel(self, a: int, b: int) -> Optional[float]:
        e = self._edges.get(frozenset((a, b)))
        return e["max"] if e else None

    def project(self, a: int, b: int, point) -> Optional[np.ndarray]:
        """Maps a floor point from the image of camera a to camera b (None if unknown)."""
        H = self._H.get((a, b))
        if H is None or point is None:
            return None
        p = H @ np.array([point[0], point[1], 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return p[:2] / p[2]

    def impossible_reason(self, sightings: Dict[int, float], camera_id: int, t: float,
                          positions: Optional[Dict[int, tuple]] = None, foot=None,
                          now: Optional[float] = None) -> Optional[str]:
        """
        Can a person whose last sighting per camera is `sightings` ({camera: time})
        appear in `camera_id` at time `t`?

        Optional location check for overlapping cameras: `positions` is the
        person's last foot point per camera, `foot` the query person's current
        foot point in `camera_id`, `now` the current time.

        Returns None if possible, otherwise a short human-readable reason.
        """
        now = t if now is None else now
        for cam, seen in sightings.items():
            if cam == camera_id:
                continue
            if self.overlaps(cam, camera_id):
                if foot is None or not positions or now - seen > self.simultaneous_window:
                    continue
                mapped = self.project(cam, camera_id, positions.get(cam))
                if mapped is not None:
                    dist = float(np.hypot(mapped[0] - foot[0], mapped[1] - foot[1]))
                    if dist > self.spatial_gate:
                        return f"standing elsewhere in Cam {cam} right now ({dist:.0f}px away)"
                continue
            gap = t - seen
            if gap < self.simultaneous_window:
                return f"visible in Cam {cam} at the same time"
            need = self.min_travel(cam, camera_id)
            if gap < need:
                return f"left Cam {cam} {gap:.1f}s ago, walking to Cam {camera_id} takes >= {need:.0f}s"
        return None

    def describe(self) -> str:
        lines = []
        if self._overlap:
            lines.append("overlapping: " + ", ".join(f"{sorted(p)}" for p in self._overlap))
        for pair, e in self._edges.items():
            a, b = sorted(pair)
            mx = f"{e['max']:.0f}" if e["max"] is not None else "-"
            lines.append(f"path Cam {a} <-> Cam {b}: min {e['min']:.0f}s, max {mx}s")
        if not lines:
            lines.append("no topology defined (only rule: one person, one place at a time)")
        return "; ".join(lines)
