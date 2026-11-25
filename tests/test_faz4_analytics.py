"""
Analytics calculations (src/analytics/stats.py) and the session recorder on small
hand-made data whose answers are known. Run:  python tests/test_faz4_analytics.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from analytics import stats  # noqa: E402
from analytics.recorder import SessionRecorder  # noqa: E402
from utils.privacy import pixelate_heads  # noqa: E402

SP = 0.5  # sample period


def walk(pid, cam, t0, t1, x0, y0, x1, y1):
    """Positions of one person walking in a straight line, one sample per 0.5 s."""
    ts = np.arange(t0, t1 + 1e-9, SP)
    f = (ts - t0) / max(1e-9, t1 - t0)
    return pd.DataFrame({"t": ts, "cam": cam, "pid": pid, "x": x0 + f * (x1 - x0), "y": y0 + f * (y1 - y0)})


def test_visits_and_transitions():
    # P1: camera 0 for 10 s, unseen 6 s, camera 1 for 20 s. P2: only camera 1.
    pos = pd.concat([walk(1, 0, 0, 10, 0, 0, 10, 0), walk(1, 1, 16, 36, 0, 0, 10, 0),
                     walk(2, 1, 0, 5, 0, 0, 1, 0)], ignore_index=True)
    v = stats.visits(pos, "cam", gap=5.0, sample_period=SP)
    p1 = v[v["pid"] == 1]
    assert list(p1["place"]) == [0, 1]
    assert list(p1["dwell"]) == [10.5, 20.5]
    tr = stats.transitions(v)
    assert len(tr) == 1 and tr.iloc[0][["src", "dst", "seconds"]].tolist() == [0, 1, 6.0]
    summary = stats.dwell_summary(v).set_index("place")
    assert summary.loc[1, "people"] == 2


def test_gap_splits_visit():
    pos = pd.concat([walk(1, 0, 0, 4, 0, 0, 1, 0), walk(1, 0, 20, 24, 0, 0, 1, 0)], ignore_index=True)
    v = stats.visits(pos, "cam", gap=5.0, sample_period=SP)
    assert len(v) == 2          # left and came back = two visits
    assert stats.transitions(v).empty  # same place: not a move


def test_overlapping_cameras_are_not_moves():
    # Seen by two cameras at the same time (same room): no transition between them
    pos = pd.concat([walk(1, 0, 0, 10, 0, 0, 1, 0), walk(1, 1, 2, 8, 0, 0, 1, 0)], ignore_index=True)
    assert stats.transitions(stats.visits(pos, "cam", 5.0, SP)).empty


def test_merges_are_applied():
    with tempfile.TemporaryDirectory() as d:
        rec = SessionRecorder([{"id": 0, "name": "door"}], label="t", clock="video", origin=0.0, root=Path(d))
        for t in np.arange(0, 3, SP):
            rec.position(5, 0, t, (1.0, 1.0))
        rec.merge(keep=3, drop=5)
        rec.merge(keep=1, drop=3)   # chain: 5 -> 3 -> 1
        rec.close()
        s = stats.load_session(rec.dir)
        assert set(s.positions["pid"]) == {1}
        assert len(s.positions) == 6


def test_recorder_throttles_samples():
    with tempfile.TemporaryDirectory() as d:
        rec = SessionRecorder([{"id": 0, "name": "a"}], clock="video", origin=0.0, root=Path(d))
        for t in np.arange(0, 2, 0.1):  # 10 fps for 2 s
            rec.position(1, 0, float(t), (0, 0))
        rec.close()
        assert len(stats.load_session(rec.dir).positions) == 4  # one per 0.5 s


def test_zones():
    cams = {0: {"name": "shop", "zones": {"till": [[0, 0], [10, 0], [10, 10], [0, 10]]}}}
    pos = pd.concat([walk(1, 0, 0, 4, 5, 5, 5, 5),          # 4 s at the till
                     walk(1, 0, 4.5, 8, 50, 50, 50, 50)], ignore_index=True)  # then elsewhere
    z = stats.zone_samples(pos, cams)
    assert set(z["zone"]) == {"shop / till"} and len(z) == 9
    v = stats.visits(z, "zone", 5.0, SP)
    assert v.iloc[0]["dwell"] == 4.5


def test_line_crossings():
    # Vertical line x = 50 from (50,0) to (50,100). Standing at A looking at B
    # (downwards in the image) the right-hand side is x < 50.
    cams = {0: {"name": "door", "lines": {"entrance": [[50, 0], [50, 100]]}}}
    pos = pd.concat([
        walk(1, 0, 0, 5, 90, 50, 10, 50),     # right to left across x=50 -> ends on x<50: "in"
        walk(2, 0, 0, 5, 10, 50, 90, 50),     # "out"
        walk(3, 0, 0, 5, 90, 150, 10, 150),   # passes below the drawn segment: not counted
        walk(4, 0, 0, 5, 52, 50, 48, 50),     # hovering on the line: not counted
    ], ignore_index=True)
    ev = stats.line_events(pos, cams)
    assert sorted(zip(ev["pid"], ev["direction"])) == [(1, "in"), (2, "out")], ev
    s = stats.line_summary(ev).iloc[0]
    assert (s["in"], s["out"], s["net"]) == (1, 1, 0)


def test_heatmap_counts_seconds():
    pos = walk(1, 0, 0, 9.5, 20, 20, 20, 20)  # 20 samples at one spot
    hm = stats.heatmap(pos, 0, (100, 100), SP, cell=10, smooth=0)
    assert hm[2, 2] == 10.0 and hm.sum() == 10.0


def test_head_pixelation():
    img = np.random.default_rng(0).integers(0, 255, (200, 100, 3), dtype=np.uint8)
    before = img.copy()
    pixelate_heads(img, [(20, 20, 80, 180)])
    assert not np.array_equal(img[25:45, 25:75], before[25:45, 25:75])  # head changed
    assert np.array_equal(img[120:180], before[120:180])                # legs untouched


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")
