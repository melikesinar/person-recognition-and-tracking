"""
Tests for the Faz 2 decision rules (topology, presence, "not sure").

They use small hand-made situations, so they run in a second without videos or
models. Run from the project root:
    python tests/test_faz2_rules.py          (or: python -m pytest tests)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.config import Config  # noqa: E402
from tracking.topology import CameraTopology  # noqa: E402
from tracking.presence import PresenceRegistry  # noqa: E402
from tracking.identity_matcher import IdentityMatcher  # noqa: E402

# Office example: door (0) -- corridor (1) -- office (2); 3 and 4 see the same room.
OFFICE = {
    "overlapping": [[3, 4]],
    "transitions": [
        {"cameras": [0, 1], "min_seconds": 3, "max_seconds": 60},
        {"cameras": [1, 2], "min_seconds": 5, "max_seconds": 60},
    ],
}


def test_topology_rules():
    topo = CameraTopology(range(5), OFFICE)
    assert topo.min_travel(0, 2) == 8              # door -> corridor -> office
    assert topo.min_travel(2, 0) == 8              # paths work both ways
    assert topo.min_travel(3, 4) == 0              # same room
    assert topo.min_travel(0, 3) == 0              # no path known -> default (0)

    # Seen at the door at t=100
    seen = {0: 100.0}
    assert topo.impossible_reason(seen, 2, 100.5) is not None   # same moment, two places
    assert topo.impossible_reason(seen, 2, 105.0) is not None   # 5 s < 8 s walking time
    assert topo.impossible_reason(seen, 2, 109.0) is None       # 9 s: possible
    assert topo.impossible_reason(seen, 0, 100.5) is None       # same camera: not this rule's job
    assert topo.impossible_reason({3: 100.0}, 4, 100.0) is None  # overlapping cameras: allowed


def test_topology_rejects_unknown_camera():
    try:
        CameraTopology([0, 1], {"transitions": [{"cameras": [0, 7], "min_seconds": 1}]})
    except ValueError:
        return
    raise AssertionError("unknown camera id should be rejected")


def test_presence_visits_transitions_merge():
    topo = CameraTopology(range(5), OFFICE)
    reg = PresenceRegistry(topo, {0: "door", 1: "corridor", 2: "office"}, visit_gap_seconds=5)
    for t in np.arange(0, 10, 0.5):
        reg.observe(1, 0, t)            # P1 at the door 0..9.5
    for t in np.arange(14, 30, 0.5):
        reg.observe(1, 1, t)            # then corridor 14..29.5
    for t in np.arange(40, 45, 0.5):
        reg.observe(2, 2, t)            # P2 (actually P1) in the office 40..44.5

    visits = reg.visits()
    assert [(v.pid, v.camera_id) for v in visits] == [(1, 0), (1, 1), (2, 2)]
    assert abs(visits[0].duration - 9.5) < 1e-6
    assert len(reg.transitions) == 1 and abs(reg.transitions[0].seconds - 4.5) < 1e-6

    reg.merge(keep_pid=1, drop_pid=2)
    assert {v.pid for v in reg.visits()} == {1}
    assert reg.sightings(1) == {0: 9.5, 1: 29.5, 2: 44.5}

    # Leaving and coming back after more than visit_gap starts a new visit
    reg.observe(3, 0, 100.0)
    reg.observe(3, 0, 110.0)
    assert len([v for v in reg.visits() if v.pid == 3]) == 2


class FakeIndex:
    def __init__(self, sims):
        self.sims = sims  # {pid: reid similarity}

    def top_persons(self, vector, k, min_sim):
        out = sorted(self.sims.items(), key=lambda kv: -kv[1])
        return [(p, s) for p, s in out if s >= min_sim][:k]


class FakePeople:
    def part_similarities(self, pid, cues):
        return {}  # no colour: fused score == OSNet score


def make_matcher(sims, topo=None, presence=None):
    return IdentityMatcher(FakePeople(), FakeIndex(sims), topo, presence)


def test_matcher_not_sure():
    Config.AMBIGUITY_MARGIN = 0.03
    q = np.zeros(4, dtype=np.float32)

    r = make_matcher({1: 0.92, 2: 0.70}).match(q, [], 0.84)
    assert r.pid == 1 and not r.ambiguous                     # clear winner

    r = make_matcher({1: 0.92, 2: 0.91}).match(q, [], 0.84)
    assert r.pid is None and r.ambiguous                       # two people, too close to call
    assert r.best.pid == 1 and r.rival.pid == 2

    r = make_matcher({1: 0.80}).match(q, [], 0.84)
    assert r.pid is None and not r.ambiguous                   # below threshold: new person

    # P1 looks best, but is in this camera already (another track): impossible
    r = make_matcher({1: 0.95, 2: 0.88}).match(q, [], 0.84, unavailable={1})
    assert r.pid is None and r.ambiguous and r.rival.pid == 1  # looks most like someone else

    r = make_matcher({1: 0.95, 2: 0.70}).match(q, [], 0.84, unavailable={1})
    assert r.pid is None and r.blocked_by.pid == 1             # wait, do not create a duplicate


def test_matcher_space_time():
    topo = CameraTopology(range(5), OFFICE)
    reg = PresenceRegistry(topo, {}, 5)
    reg.observe(1, 0, 100.0)  # P1 at the door at t=100
    reg.observe(2, 1, 50.0)   # P2 in the corridor long ago
    q = np.zeros(4, dtype=np.float32)

    # Two same-dressed people; appearance slightly favours P1, but P1 cannot be in
    # the office 3 s after being at the door. The system must not pick P1, and
    # because the impossible P1 looks better than P2 it says "not sure".
    m = make_matcher({1: 0.93, 2: 0.91}, topo, reg)
    r = m.match(q, [], 0.84, camera_id=2, t=103.0)
    assert r.pid is None and r.rival.pid == 1 and "walking" in r.rival.impossible

    # Appearance clearly says P2 and P1 is impossible: P2 is accepted only if P1 is not
    # better looking. Here P2 is better -> decided.
    m = make_matcher({1: 0.88, 2: 0.94}, topo, reg)
    r = m.match(q, [], 0.84, camera_id=2, t=103.0)
    assert r.pid == 2

    # 20 s later P1 could have walked there: normal decision
    m = make_matcher({1: 0.93, 2: 0.70}, topo, reg)
    assert m.match(q, [], 0.84, camera_id=2, t=120.0).pid == 1


def test_same_spot_rule_for_overlapping_cameras():
    # Cameras 0 and 1 see the same room; the floor mapping is a simple shift by +100 px
    H = [[1, 0, 100], [0, 1, 0], [0, 0, 1]]
    topo = CameraTopology([0, 1], {"overlapping": [[0, 1]]}, homographies={"0-1": H}, spatial_gate_px=60)
    seen, pos = {0: 50.0}, {0: (200.0, 300.0)}   # P is at (200,300) in Cam 0 -> (300,300) in Cam 1
    assert topo.impossible_reason(seen, 1, 50.0, pos, foot=(310.0, 300.0), now=50.2) is None
    assert "elsewhere" in topo.impossible_reason(seen, 1, 50.0, pos, foot=(100.0, 300.0), now=50.2)
    # Inverse direction is derived automatically
    assert topo.impossible_reason({1: 50.0}, 0, 50.0, {1: (300.0, 300.0)}, foot=(205.0, 300.0), now=50.2) is None
    # P was seen in Cam 0 long ago: where they stood then says nothing about now
    assert topo.impossible_reason(seen, 1, 60.0, pos, foot=(100.0, 300.0), now=60.0) is None


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")
