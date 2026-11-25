"""
Statistics from a recorded session (Faz 4). Pure functions on pandas tables, used
by the dashboard and tested in tests/test_faz4_analytics.py.

Input: a session folder written by analytics/recorder.py. The positions table
(t, cam, pid, x, y) is the single source of truth; everything else is derived
from it:
    visits        a person's continuous presence in a camera or zone
                  (a gap longer than `gap` seconds starts a new visit)
    transitions   consecutive visits of one person in two different places
    occupancy     distinct people per camera per time bin
    heat map      seconds spent per floor spot (image pixels) per camera
    line events   crossings of a counting line, with direction (in / out)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

POS_COLUMNS = ["t", "cam", "pid", "x", "y"]


# --------------------------------------------------------------------- loading

@dataclass
class Session:
    path: Path
    meta: dict
    positions: pd.DataFrame
    health: pd.DataFrame
    merges: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def cameras(self) -> Dict[int, dict]:
        return {int(c["id"]): c for c in self.meta.get("cameras", [])}

    def cam_name(self, cam: int) -> str:
        return self.cameras.get(int(cam), {}).get("name", f"cam{cam}")

    @property
    def is_video(self) -> bool:
        return self.meta.get("clock") == "video"

    def rel(self, t):
        """Time relative to the session start (video) or unchanged (wall clock)."""
        return t - self.meta["origin"] if self.is_video and self.meta.get("origin") is not None else t

    @property
    def last_time(self) -> Optional[float]:
        if self.positions.empty:
            return None
        return float(self.positions["t"].max())


def list_sessions(root: Path) -> List[Path]:
    """Session folders, newest first."""
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if (d / "meta.json").is_file()),
                  key=lambda d: d.name, reverse=True)


def _read_csv(path: Path, columns: List[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)
    return df.dropna() if not df.empty else pd.DataFrame(columns=columns)


def resolve_merges(merges: pd.DataFrame) -> Dict[int, int]:
    """{dropped id: final id}, following chains (P5 -> P3, later P3 -> P1 gives P5 -> P1)."""
    parent: Dict[int, int] = {}

    def find(x):
        while x in parent:
            x = parent[x]
        return x

    for _, row in merges.iterrows():
        keep, drop = find(int(row["keep"])), find(int(row["drop"]))
        if keep != drop:
            parent[drop] = keep
    return {k: find(k) for k in parent}


def load_session(path: Path) -> Session:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    pos = _read_csv(path / "positions.csv", POS_COLUMNS)
    merges = _read_csv(path / "merges.csv", ["t", "keep", "drop"])
    health = _read_csv(path / "health.csv", ["t", "cam", "fps", "target_fps", "status"])
    pos = pos.astype({"t": float, "cam": int, "pid": int, "x": float, "y": float}) if not pos.empty else pos
    mapping = resolve_merges(merges) if not merges.empty else {}
    if mapping and not pos.empty:
        pos["pid"] = pos["pid"].map(lambda p: mapping.get(p, p))
    return Session(path, meta, pos.sort_values("t").reset_index(drop=True) if not pos.empty else pos,
                   health, merges)


# --------------------------------------------------------------------- geometry

def points_in_polygon(xy: np.ndarray, poly) -> np.ndarray:
    """Boolean array: which of the (N, 2) points are inside the polygon (ray casting)."""
    poly = np.asarray(poly, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_at)
        j = i
    return inside


def signed_distance(xy: np.ndarray, a, b) -> np.ndarray:
    """Signed distance of points to the line a->b; positive on the line's "in" side."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = np.hypot(*d) or 1.0
    return (d[0] * (xy[:, 1] - a[1]) - d[1] * (xy[:, 0] - a[0])) / n


def _segments_intersect(p, q, a, b) -> bool:
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2 = cross(a, b, p), cross(a, b, q)
    d3, d4 = cross(p, q, a), cross(p, q, b)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


# --------------------------------------------------------------------- places

def zone_samples(pos: pd.DataFrame, cameras: Dict[int, dict]) -> pd.DataFrame:
    """One row per (position sample, zone it falls in), with a `zone` column 'camera / zone'."""
    rows = []
    for cam, info in cameras.items():
        sub = pos[pos["cam"] == cam]
        if sub.empty:
            continue
        xy = sub[["x", "y"]].to_numpy()
        for name, poly in (info.get("zones") or {}).items():
            inside = sub[points_in_polygon(xy, poly)]
            if not inside.empty:
                rows.append(inside.assign(zone=f"{info.get('name', cam)} / {name}"))
    if not rows:
        return pd.DataFrame(columns=POS_COLUMNS + ["zone"])
    return pd.concat(rows, ignore_index=True)


def visits(pos: pd.DataFrame, place: str, gap: float, sample_period: float) -> pd.DataFrame:
    """
    Continuous stays of each person in each place (`place` = column name: "cam" or "zone").
    dwell counts each sample as `sample_period` seconds, so a single sighting is not 0 s.
    """
    cols = ["pid", "place", "start", "end", "dwell"]
    if pos.empty:
        return pd.DataFrame(columns=cols)
    df = pos.sort_values(["pid", place, "t"])
    new = (df["pid"] != df["pid"].shift()) | (df[place] != df[place].shift()) | (df["t"].diff() > gap)
    df = df.assign(visit=new.cumsum())
    out = df.groupby("visit").agg(pid=("pid", "first"), place=(place, "first"),
                                  start=("t", "min"), end=("t", "max"))
    out["dwell"] = out["end"] - out["start"] + sample_period
    return out.sort_values(["pid", "start"]).reset_index(drop=True)


def dwell_summary(v: pd.DataFrame) -> pd.DataFrame:
    """Per place: visits, distinct people, average / median / longest stay (seconds)."""
    if v.empty:
        return pd.DataFrame(columns=["place", "visits", "people", "avg_s", "median_s", "max_s", "total_s"])
    g = v.groupby("place")
    return pd.DataFrame({
        "visits": g.size(), "people": g["pid"].nunique(), "avg_s": g["dwell"].mean(),
        "median_s": g["dwell"].median(), "max_s": g["dwell"].max(), "total_s": g["dwell"].sum(),
    }).reset_index().sort_values("total_s", ascending=False)


def transitions(v: pd.DataFrame) -> pd.DataFrame:
    """
    A person's move from one place to the next: consecutive visits (by start time)
    in different places. Visits that overlap in time (two cameras seeing the same
    spot) are not moves and are skipped.
    """
    cols = ["pid", "src", "dst", "seconds", "t"]
    if v.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for pid, grp in v.sort_values("start").groupby("pid"):
        prev = None
        for r in grp.itertuples():
            if prev is not None and r.place != prev.place and r.start >= prev.end:
                rows.append((pid, prev.place, r.place, r.start - prev.end, r.start))
            if prev is None or r.end >= prev.end:
                prev = r
    return pd.DataFrame(rows, columns=cols)


def transition_summary(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame(columns=["src", "dst", "count", "median_s", "min_s", "max_s"])
    g = tr.groupby(["src", "dst"])["seconds"]
    return pd.DataFrame({"count": g.size(), "median_s": g.median(), "min_s": g.min(),
                         "max_s": g.max()}).reset_index().sort_values("count", ascending=False)


def routes(v: pd.DataFrame, name=lambda p: str(p), min_dwell: float = 1.0) -> pd.DataFrame:
    """One row per person: first/last seen, route text 'door 8s -> corridor 41s', time in view."""
    cols = ["pid", "first_seen", "last_seen", "places", "route", "time_in_view_s"]
    if v.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for pid, grp in v[v["dwell"] >= min_dwell].sort_values("start").groupby("pid"):
        route = " → ".join(f"{name(r.place)} {r.dwell:.0f}s" for r in grp.itertuples())
        rows.append((pid, grp["start"].min(), grp["end"].max(), grp["place"].nunique(), route,
                     grp["dwell"].sum()))
    return pd.DataFrame(rows, columns=cols).sort_values("first_seen").reset_index(drop=True)


# --------------------------------------------------------------------- time

def occupancy(pos: pd.DataFrame, bin_seconds: float) -> pd.DataFrame:
    """Distinct people per camera per time bin: columns bin_start, cam, people."""
    if pos.empty:
        return pd.DataFrame(columns=["bin_start", "cam", "people"])
    b = (np.floor(pos["t"] / bin_seconds) * bin_seconds).rename("bin_start")
    out = pos.groupby([b, pos["cam"]])["pid"].nunique().rename("people").reset_index()
    return out


def visible_now(pos: pd.DataFrame, now: float, window: float = 3.0) -> Dict[int, List[int]]:
    """{camera: [people seen in the last `window` seconds]}."""
    recent = pos[pos["t"] >= now - window] if not pos.empty else pos
    return {int(c): sorted(int(p) for p in g["pid"].unique()) for c, g in recent.groupby("cam")} \
        if not recent.empty else {}


# --------------------------------------------------------------------- heat map

def heatmap(pos: pd.DataFrame, cam: int, size, sample_period: float, cell: int = 8,
            smooth: float = 2.0) -> np.ndarray:
    """
    Seconds spent per image area in camera `cam`: a (H/cell, W/cell) grid where
    each foot sample adds `sample_period` seconds, then Gaussian-smoothed.
    """
    w, h = int(size[0]), int(size[1])
    grid = np.zeros((max(1, h // cell), max(1, w // cell)), dtype=np.float64)
    sub = pos[pos["cam"] == cam] if not pos.empty else pos
    if sub.empty:
        return grid
    gx = np.clip((sub["x"].to_numpy() // cell).astype(int), 0, grid.shape[1] - 1)
    gy = np.clip((sub["y"].to_numpy() // cell).astype(int), 0, grid.shape[0] - 1)
    np.add.at(grid, (gy, gx), sample_period)
    if smooth > 0:
        import cv2
        grid = cv2.GaussianBlur(grid, (0, 0), smooth)
    return grid


# --------------------------------------------------------------------- counting lines

def line_events(pos: pd.DataFrame, cameras: Dict[int, dict], margin: float = 6.0,
                max_gap: float = 3.0) -> pd.DataFrame:
    """
    Crossings of each counting line: t, cam, line, pid, direction ("in"/"out").

    A line is [A, B] in image pixels. Crossing it towards its right-hand side (as
    seen standing on A looking at B, in the image) is "in", towards the left-hand
    side "out"; tools/draw_zones.py marks the "in" side with an arrow.
    Noise control: a person's side only counts once they are more than `margin`
    pixels from the line (no double counts when standing on it), the move must
    cross the drawn segment (not its extension), and position samples more than
    `max_gap` seconds apart are not joined (the person was out of view).
    """
    cols = ["t", "cam", "line", "pid", "direction"]
    rows = []
    for cam, info in cameras.items():
        lines = info.get("lines") or {}
        if not lines or pos.empty:
            continue
        sub = pos[pos["cam"] == cam].sort_values(["pid", "t"])
        for name, (a, b) in lines.items():
            for pid, grp in sub.groupby("pid"):
                xy = grp[["x", "y"]].to_numpy()
                ts = grp["t"].to_numpy()
                d = signed_distance(xy, a, b)
                last_side, last_i = 0, None
                for i in range(len(xy)):
                    if last_i is not None and ts[i] - ts[i - 1] > max_gap:
                        last_side, last_i = 0, None
                    side = 1 if d[i] > margin else (-1 if d[i] < -margin else 0)
                    if side == 0:
                        continue
                    if last_side and side != last_side and _segments_intersect(xy[last_i], xy[i], a, b):
                        rows.append((ts[i], cam, name, pid, "in" if side > 0 else "out"))
                    last_side, last_i = side, i
    return pd.DataFrame(rows, columns=cols).sort_values("t").reset_index(drop=True) if rows \
        else pd.DataFrame(columns=cols)


def line_summary(ev: pd.DataFrame) -> pd.DataFrame:
    """Per line: entries, exits, and net = currently inside (if everyone passes this line)."""
    if ev.empty:
        return pd.DataFrame(columns=["cam", "line", "in", "out", "net"])
    c = ev.groupby(["cam", "line", "direction"]).size().unstack(fill_value=0)
    for k in ("in", "out"):
        if k not in c:
            c[k] = 0
    c["net"] = c["in"] - c["out"]
    return c[["in", "out", "net"]].reset_index()
