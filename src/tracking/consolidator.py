"""
Identity consolidation (Faz 3): finds identities that are really the same person
(one person split into several P numbers) and merges them.

Why it is needed: Faz 2 made the system careful. When unsure it waits, and when
it waits too long it opens a new identity rather than guess. That keeps routes
clean but splits some people into two or three numbers. Splits are recoverable:
once more evidence has accumulated (bigger galleries, more colour samples), the
same question can be asked again with much more information than the first
decision had.

Every CONSOLIDATE_INTERVAL_SECONDS, for every pair of identities (A, B):
    1. Never merge if they are proven different people (seen at the same time in
       the same camera, in two places at once, ...; PresenceRegistry keeps this
       "cannot-link" list, it is learned on every frame).
    2. Appearance score over the WHOLE galleries, not single frames:
         OSNet: for each vector of B its best match in A, averaged (and the
                same from A to B). Robust against one lucky look-alike frame.
         Colour: per body part, the same robust average over colour samples.
       Fused with the same weights and colour veto as matching (Faz 1).
    3. Merge only if the score is high AND decisive (AMBIGUITY_MARGIN):
         look-alikes: no identity proven different from A (or B) may look about
             as similar to it as B (or A) does; otherwise this level of
             similarity does not mean "same person" for these two;
         rivals: B must not look about as similar to some C that is proven
             different from A (B could equally be C's fragment), and vice versa.
At most CONSOLIDATE_MAX_MERGES merges per round, best first; the rest are
re-evaluated next round with the merged galleries.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.config import Config
from utils.logger import get_logger
from core.appearance_cues import PART_NAMES, fuse_scores

logger = get_logger()


class IdentityConsolidator:
    def __init__(self, person_manager, presence, topology):
        self.pm = person_manager
        self.presence = presence
        self.topology = topology
        self.last_run: Optional[float] = None

    # --- Called every processed time step --------------------------------------

    def step(self, camera_nodes, now: float):
        visible = {}
        for node in camera_nodes:
            bm = node.buffer_manager
            people = {}
            for pid in bm.track_id_to_permanent_id_map.values():
                box = bm.last_known_box_for_pid.get(pid)
                people[pid] = None if box is None else ((float(box[0]) + float(box[2])) / 2.0, float(box[3]))
            visible[node.camera_id] = people
        self.presence.note_simultaneous(visible, self.topology.spatial_gate)

        if not Config.CONSOLIDATION_ENABLED:
            return
        if self.last_run is not None and now - self.last_run < Config.CONSOLIDATE_INTERVAL_SECONDS:
            return
        self.last_run = now
        for keep, drop, info in self.find_merges()[:Config.CONSOLIDATE_MAX_MERGES]:
            if keep in self.pm.known_persons_db and drop in self.pm.known_persons_db:
                logger.info(f"CONSOLIDATE: P{drop} -> P{keep} {info}")
                merge_everywhere(keep, drop, self.pm, self.presence, camera_nodes)

    # --- Scoring -----------------------------------------------------------------

    @staticmethod
    def _gallery_similarity(x: np.ndarray, y: np.ndarray) -> float:
        """Each sample's best match on the other side, averaged both ways (robust to one lucky frame)."""
        s = x @ y.T
        return 0.5 * (float(s.max(axis=1).mean()) + float(s.max(axis=0).mean()))

    def _galleries(self, pid: int) -> dict:
        d = self.pm.known_persons_db[pid]
        g = {"reid": np.asarray(d["vectors"], dtype=np.float32)}
        for name in PART_NAMES:
            samples = [c.parts[name] for c in d.get("cues", []) if c.parts.get(name) is not None]
            g[name] = np.stack(samples).astype(np.float32) if samples else None
        return g

    def pair_score(self, ga: dict, gb: dict) -> Tuple[float, float, Dict[str, float]]:
        """(fused, osnet, colour parts) similarity between two whole identities; fused=-1 if vetoed."""
        reid = self._gallery_similarity(ga["reid"], gb["reid"])
        parts = {name: self._gallery_similarity(ga[name], gb[name])
                 for name in PART_NAMES if ga[name] is not None and gb[name] is not None}
        fused, vetoed = fuse_scores(reid, parts)
        return (-1.0 if vetoed else fused), reid, parts

    def find_merges(self) -> List[Tuple[int, int, str]]:
        """Returns [(keep, drop, explanation)] best first, without overlapping ids."""
        ids = [p for p, d in self.pm.known_persons_db.items()
               if len(d.get("vectors", [])) >= Config.CONSOLIDATE_MIN_VECTORS]
        if len(ids) < 2:
            return []

        galleries = {p: self._galleries(p) for p in ids}
        scores: Dict[Tuple[int, int], Tuple[float, float, Dict[str, float]]] = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                scores[(ids[i], ids[j])] = self.pair_score(galleries[ids[i]], galleries[ids[j]])

        def score(x, y):
            return scores[(min(x, y), max(x, y))][0]

        diff = self.presence.proven_different
        margin = Config.AMBIGUITY_MARGIN
        proposals = []
        for (a, b), (s, reid, parts) in scores.items():
            if s < Config.CONSOLIDATE_MIN_SCORE or diff(a, b):
                continue
            others = [c for c in ids if c not in (a, b)]
            # (1) Calibration by look-alikes: people proven different from a (or b)
            # may not look as similar to it as b (or a) does. Then this level of
            # similarity does not mean "same person" for these two.
            lookalike = max((score(c, x) for x in (a, b) for c in others if diff(c, x)), default=-1.0)
            # (2) Competition: b could equally be a fragment of c, where c is a
            # different person from a (and vice versa). Then merging is a coin toss.
            rival = max([score(c, b) for c in others if diff(c, a) and not diff(c, b)] +
                        [score(c, a) for c in others if diff(c, b) and not diff(c, a)], default=-1.0)
            if lookalike > s - margin or rival > s - margin:
                continue
            colours = ", ".join(f"{k}={v:.2f}" for k, v in parts.items()) or "no colour"
            info = (f"fused={s:.2f} (reid={reid:.2f}; {colours}); best look-alike {lookalike:.2f}, "
                    f"best rival {rival:.2f}")
            proposals.append((s, min(a, b), max(a, b), info))

        out, used = [], set()
        for _, keep, drop, info in sorted(proposals, key=lambda p: -p[0]):
            if keep in used or drop in used or (keep, drop) in {(k, d) for k, d, _ in out}:
                continue
            out.append((keep, drop, info))
            used.update((keep, drop))
        return out


def merge_everywhere(keep: int, drop: int, person_manager, presence, camera_nodes):
    """
    Merges identity `drop` into `keep`: galleries, history, and every live
    reference in every camera (tracks currently showing P{drop} now show P{keep}).
    """
    person_manager.merge_identities(keep_pid=keep, drop_pid=drop)
    if presence is not None:
        presence.merge(keep, drop)
    for node in camera_nodes:
        bm = node.buffer_manager
        for tid, pid in list(bm.track_id_to_permanent_id_map.items()):
            if pid == drop:
                bm.track_id_to_permanent_id_map[tid] = keep
        for store in (bm.last_known_box_for_pid, bm.lost_pid_buffer, bm.last_update_time_for_pid):
            if drop in store:
                value = store.pop(drop)
                store.setdefault(keep, value)
        for excluded in getattr(node, "seen_with", {}).values():
            if drop in excluded:
                excluded.discard(drop)
                excluded.add(keep)
