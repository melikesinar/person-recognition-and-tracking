"""
Checks the identity metrics (tools/id_metrics.py) on tiny hand-made cases whose
correct answers can be worked out by hand.
Run:  python tests/test_id_metrics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from id_metrics import evaluate  # noqa: E402

A_BOX = (0, 0, 10, 20)
B_BOX = (50, 0, 60, 20)


def scene(pred_ids, frames=4, cams=(0, 1)):
    """Persons 'a' and 'b' visible in every camera/frame; pred_ids(cam, frame) -> (id_a, id_b)."""
    gt, pred = {}, {}
    for c in cams:
        for f in range(frames):
            gt[(c, f)] = [("a", A_BOX), ("b", B_BOX)]
            ia, ib = pred_ids(c, f)
            pred[(c, f)] = [(ia, (0.5, 0, 10.5, 20)), (ib, (50, 0.5, 60, 20.5))]  # slightly shifted
    return gt, pred


def close(x, y):
    return abs(x - y) < 1e-9


def test_perfect():
    r = evaluate(*scene(lambda c, f: (1, 2)))
    assert close(r["global"]["IDF1"], 1.0) and close(r["detected"], 1.0)
    assert r["fragmentation"] == 1.0 and r["mixing"] == 1.0


def test_inconsistent_across_cameras():
    # Within each camera the numbers are stable, but b is P2 in cam 0 and P3 in cam 1
    r = evaluate(*scene(lambda c, f: (1, 2 if c == 0 else 3)))
    assert close(r["within_camera"]["IDF1"], 1.0)
    # 16 boxes; b's 4 boxes in cam 1 cannot be identity-correct globally
    assert close(r["global"]["IDF1"], 12 / 16)
    assert r["fragmentation"] == 1.5  # a: 1 number, b: 2 numbers


def test_swap_halfway():
    # In both cameras the two people swap numbers after frame 1
    r = evaluate(*scene(lambda c, f: (1, 2) if f < 2 else (2, 1)))
    assert close(r["global"]["IDF1"], 0.5)
    assert r["mixing"] == 2.0


def test_unknown_and_missed():
    gt, pred = scene(lambda c, f: (1, None))     # b never gets a number
    pred[(0, 0)] = pred[(0, 0)][:1]               # b missed by the detector once
    pred[(1, 3)].append((7, (200, 200, 210, 220)))  # someone outside the labelled area
    r = evaluate(gt, pred)
    assert close(r["detected"], 15 / 16)
    assert r["unmatched"] == 1
    # IDTP = 8 (all of a); numbered matched boxes = 8; GT = 16
    assert close(r["global"]["IDR"], 0.5) and close(r["global"]["IDP"], 1.0)
    assert close(r["global"]["IDF1"], 16 / 24)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")
