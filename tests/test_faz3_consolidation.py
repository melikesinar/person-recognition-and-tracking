"""
Decision rules of identity consolidation (src/tracking/consolidator.py) on small
hand-made galleries. Run:  python tests/test_faz3_consolidation.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.config import Config  # noqa: E402
from tracking.topology import CameraTopology  # noqa: E402
from tracking.presence import PresenceRegistry  # noqa: E402
from tracking.consolidator import IdentityConsolidator  # noqa: E402

rng = np.random.default_rng(0)
BASE = {name: rng.normal(size=64) for name in ("alice", "bob")}


def gallery(person, n=6, noise=0.25):
    """n unit vectors around a person's base appearance."""
    vs = BASE[person] + noise * rng.normal(size=(n, 64)) * np.linalg.norm(BASE[person]) / 8
    return [v / np.linalg.norm(v) for v in vs.astype(np.float32)]


class FakePM:
    def __init__(self, db):
        self.known_persons_db = db


def make(db, cams=(0, 1)):
    topo = CameraTopology(cams, {})
    presence = PresenceRegistry(topo, {}, 5)
    return IdentityConsolidator(FakePM(db), presence, topo), presence


def setup_module():
    Config.CUE_WEIGHTS = {"reid": 1.0, "upper": 0.0, "lower": 0.0, "head": 0.0}
    Config.CONSOLIDATE_MIN_SCORE = 0.80
    Config.CONSOLIDATE_MIN_VECTORS = 3
    Config.AMBIGUITY_MARGIN = 0.03


def test_fragment_is_merged():
    db = {1: {"vectors": gallery("alice")}, 2: {"vectors": gallery("bob")}, 3: {"vectors": gallery("alice")}}
    cons, _ = make(db)
    merges = cons.find_merges()
    assert [(k, d) for k, d, _ in merges] == [(1, 3)], merges


def test_proven_different_is_never_merged():
    db = {1: {"vectors": gallery("alice")}, 3: {"vectors": gallery("alice")}}
    cons, presence = make(db)
    presence.note_simultaneous({0: {1: (10, 10), 3: (80, 10)}}, gate_px=50)  # seen side by side in cam 0
    assert presence.proven_different(1, 3)
    assert cons.find_merges() == []


def test_two_people_in_two_places_at_once():
    cons, presence = make({}, cams=(0, 1))  # cameras 0 and 1 do not overlap
    presence.note_simultaneous({0: {1: (10, 10)}, 1: {2: (10, 10)}}, gate_px=50)
    assert presence.proven_different(1, 2)


def test_lookalike_is_not_merged():
    # P1 and P2 are proven different people who look alike; P3 looks like both.
    # P3 could be a fragment of either one: merging it would be a coin toss.
    db = {1: {"vectors": gallery("alice")}, 2: {"vectors": gallery("alice")}, 3: {"vectors": gallery("alice")}}
    cons, presence = make(db)
    presence.add_different(1, 2)
    assert cons.find_merges() == []


def test_merge_carries_cannot_link():
    cons, presence = make({})
    presence.add_different(3, 9)
    presence.merge(keep_pid=1, drop_pid=3)
    assert presence.proven_different(1, 9) and not presence.proven_different(3, 9)


if __name__ == "__main__":
    setup_module()
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")
