"""
Where and when each identity has been seen (Faz 2).

Two jobs:
    1. Constraint input: the last sighting time of each person in each camera,
       used by CameraTopology to rule out physically impossible matches.
    2. The actual product output: each person's route as a list of visits
       (camera, entry time, exit time, dwell time), plus the observed travel times
       between cameras, which help set the topology in the cameras YAML.

Times are plain floats from time.time(). In tools/replay.py that clock is the
video clock, so reported times are video seconds.
"""
import csv
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger()


@dataclass
class Visit:
    pid: int
    camera_id: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Transition:
    pid: int
    from_cam: int
    to_cam: int
    seconds: float


class PresenceRegistry:
    def __init__(self, topology, camera_names: Dict[int, str], visit_gap_seconds: float = 5.0):
        """
        Args:
            topology: CameraTopology (to know which camera pairs overlap).
            camera_names: {camera_id: name} for reports.
            visit_gap_seconds: a person unseen in a camera for longer than this has
                left it; seeing them again there starts a new visit.
        """
        self.topology = topology
        self.camera_names = dict(camera_names)
        self.visit_gap = float(visit_gap_seconds)
        self._lock = threading.Lock()
        self._last_seen: Dict[int, Dict[int, float]] = {}       # pid -> {camera: time}
        self._last_pos: Dict[int, Dict[int, tuple]] = {}        # pid -> {camera: foot point}
        self._open: Dict[Tuple[int, int], Visit] = {}            # (pid, camera) -> visit
        self._closed: List[Visit] = []
        self.transitions: List[Transition] = []

    # --- Updates ---------------------------------------------------------------

    def observe(self, pid: int, camera_id: int, t: float, foot: Optional[tuple] = None):
        """Record that `pid` is visible in `camera_id` at time `t` (feet at `foot` in that image)."""
        with self._lock:
            key = (pid, camera_id)
            visit = self._open.get(key)
            if visit is not None and t - visit.end > self.visit_gap:
                self._closed.append(self._open.pop(key))
                visit = None
            if visit is None:
                self._record_transition(pid, camera_id, t)
                self._open[key] = Visit(pid, camera_id, t, t)
            else:
                visit.end = max(visit.end, t)
            self._last_seen.setdefault(pid, {})[camera_id] = t
            if foot is not None:
                self._last_pos.setdefault(pid, {})[camera_id] = foot

    def _record_transition(self, pid: int, camera_id: int, t: float):
        """A new visit starts: if the person came from a non-overlapping camera, log the trip."""
        prev = None
        for v in list(self._open.values()) + self._closed:
            if v.pid == pid and v.camera_id != camera_id and v.end <= t:
                if prev is None or v.end > prev.end:
                    prev = v
        if prev is None or self.topology.overlaps(prev.camera_id, camera_id):
            return
        tr = Transition(pid, prev.camera_id, camera_id, t - prev.end)
        self.transitions.append(tr)
        logger.info(f"TRANSITION: P{pid} {self._name(tr.from_cam)} -> {self._name(tr.to_cam)} "
                    f"in {tr.seconds:.1f}s")

    def close_stale(self, now: float):
        """Close visits of people who have not been seen for visit_gap seconds."""
        with self._lock:
            for key in [k for k, v in self._open.items() if now - v.end > self.visit_gap]:
                self._closed.append(self._open.pop(key))

    def merge(self, keep_pid: int, drop_pid: int):
        """Two identities turned out to be the same person: move drop_pid's history to keep_pid."""
        with self._lock:
            drop_seen = self._last_seen.pop(drop_pid, {})
            keep_seen = self._last_seen.setdefault(keep_pid, {})
            for cam, t in drop_seen.items():
                if t >= keep_seen.get(cam, t):
                    keep_seen[cam] = t
                    pos = self._last_pos.get(drop_pid, {}).get(cam)
                    if pos is not None:
                        self._last_pos.setdefault(keep_pid, {})[cam] = pos
            self._last_pos.pop(drop_pid, None)
            for v in self._closed:
                if v.pid == drop_pid:
                    v.pid = keep_pid
            for (pid, cam) in [k for k in self._open if k[0] == drop_pid]:
                v = self._open.pop((pid, cam))
                v.pid = keep_pid
                other = self._open.get((keep_pid, cam))
                if other is None:
                    self._open[(keep_pid, cam)] = v
                else:
                    other.start, other.end = min(other.start, v.start), max(other.end, v.end)
            for tr in self.transitions:
                if tr.pid == drop_pid:
                    tr.pid = keep_pid

    # --- Queries ---------------------------------------------------------------

    def sightings(self, pid: int) -> Dict[int, float]:
        with self._lock:
            return dict(self._last_seen.get(pid, {}))

    def positions(self, pid: int) -> Dict[int, tuple]:
        with self._lock:
            return dict(self._last_pos.get(pid, {}))

    def visits(self) -> List[Visit]:
        with self._lock:
            return sorted(self._closed + list(self._open.values()), key=lambda v: (v.pid, v.start))

    # --- Reports ---------------------------------------------------------------

    def _name(self, cam: int) -> str:
        return self.camera_names.get(cam, f"cam{cam}")

    @staticmethod
    def _fmt_time(t: float, origin: Optional[float]) -> str:
        if origin is not None:
            return f"{t - origin:.1f}"
        return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")

    def export_csv(self, path: Path, origin: Optional[float] = None, min_duration: float = 0.0) -> int:
        """
        Writes one row per visit. With `origin`, times are seconds since origin
        (used for recorded videos); otherwise wall-clock date-times.
        Returns the number of rows written.
        """
        rows = [v for v in self.visits() if v.duration >= min_duration]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["person", "camera_id", "camera", "enter", "exit", "dwell_seconds"])
            for v in rows:
                w.writerow([f"P{v.pid}", v.camera_id, self._name(v.camera_id),
                            self._fmt_time(v.start, origin), self._fmt_time(v.end, origin),
                            f"{v.duration:.1f}"])
        return len(rows)

    def summary_lines(self, origin: Optional[float] = None, min_duration: float = 1.0) -> List[str]:
        """Human-readable route per person, e.g. 'P3: door 12s -> corridor 40s'."""
        by_pid: Dict[int, List[Visit]] = {}
        for v in self.visits():
            if v.duration >= min_duration:
                by_pid.setdefault(v.pid, []).append(v)
        lines = []
        for pid, vs in sorted(by_pid.items()):
            route = " -> ".join(f"{self._name(v.camera_id)} {v.duration:.0f}s" for v in vs)
            lines.append(f"P{pid} (from {self._fmt_time(vs[0].start, origin)}): {route}")

        stats: Dict[Tuple[int, int], List[float]] = {}
        for tr in self.transitions:
            stats.setdefault((tr.from_cam, tr.to_cam), []).append(tr.seconds)
        for (a, b), secs in sorted(stats.items()):
            secs = sorted(secs)
            lines.append(f"travel {self._name(a)} -> {self._name(b)}: n={len(secs)}, "
                         f"min {secs[0]:.1f}s, median {secs[len(secs) // 2]:.1f}s, max {secs[-1]:.1f}s")
        return lines
