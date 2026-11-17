"""
Multi-cue identity matching (Faz 1) with space-time constraints and an
"I am not sure" outcome (Faz 2).

Decision steps:
    1. Shortlist: OSNet similarity against every stored person (best vector of
       each person's gallery) gives the top MATCH_CANDIDATES people.
    2. Re-score: each candidate gets a fused score that also uses the per-part
       clothing colour similarity (see core/appearance_cues.fuse_scores). Parts
       that are not visible are left out. A candidate whose top AND bottom colours
       clearly differ is vetoed.
    3. Feasibility (Faz 2): a candidate who is visible elsewhere at the same time,
       or could not have walked here in time (CameraTopology), is marked
       impossible. Impossible candidates can never be chosen.
    4. Decisiveness (Faz 2): the best possible candidate must be above the
       threshold AND clearly ahead of its rivals:
         * every other possible candidate must score at least AMBIGUITY_MARGIN
           lower;
         * no impossible candidate may score higher. (The person looks most like
           someone who cannot be here: maybe a look-alike, maybe a new person.)
       Otherwise the result is "ambiguous": no identity is assigned now and the
       caller waits for more evidence. A wrong identity corrupts two people's
       routes; a short "unknown" only leaves a small gap.

Keeping OSNet as the shortlist means colour alone can never pull in a stranger
who happens to wear the same colours.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from utils.config import Config
from core.appearance_cues import PartCues, fuse_scores


@dataclass
class Candidate:
    pid: int
    score: float
    reid_sim: float
    part_sims: Dict[str, float]
    impossible: Optional[str] = None  # reason, if this person cannot be here


@dataclass
class MatchResult:
    pid: Optional[int]                 # decided identity, or None
    score: float
    reid_sim: float = 0.0
    part_sims: Dict[str, float] = field(default_factory=dict)
    ambiguous: bool = False            # a match was close but not decisive
    best: Optional[Candidate] = None   # best possible candidate (even if not accepted)
    rival: Optional[Candidate] = None  # why it was not decisive
    blocked_by: Optional[Candidate] = None  # strongest match overall, if it is impossible
    candidates: List[Candidate] = field(default_factory=list)  # all scored candidates, best first

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v:.2f}" for k, v in self.part_sims.items()) or "no colour"
        return f"fused={self.score:.2f} (reid={self.reid_sim:.2f}; {parts})"

    def explain(self) -> str:
        """One-line reason for an undecided result (for logs)."""
        if self.ambiguous and self.best and self.rival:
            how = f"impossible: {self.rival.impossible}" if self.rival.impossible else "possible"
            return (f"P{self.best.pid} {self.best.score:.2f} vs P{self.rival.pid} "
                    f"{self.rival.score:.2f} ({how})")
        if self.blocked_by:
            return f"looks like P{self.blocked_by.pid} {self.blocked_by.score:.2f} but {self.blocked_by.impossible}"
        return f"best {self.score:.2f}"


class IdentityMatcher:
    def __init__(self, person_manager, vector_index, topology=None, presence=None):
        self.person_manager = person_manager
        self.vector_index = vector_index
        self.topology = topology
        self.presence = presence

    def match(self, vector: np.ndarray, cues: List[PartCues], threshold: float,
              camera_id: Optional[int] = None, t: Optional[float] = None,
              unavailable: Iterable[int] = (), foot=None, now: Optional[float] = None) -> MatchResult:
        """
        Args:
            vector: OSNet embedding of the query (single frame or averaged).
            cues: colour cue observations of the query (one or more frames).
            threshold: minimum fused score to accept.
            camera_id, t: where and since when the query person is visible
                (the track's first appearance). Needed for space-time checks.
            unavailable: identities held by other people in this camera right now.
            foot, now: the query person's current foot point in this camera and the
                current time, for the same-spot check between overlapping cameras.
        """
        unavailable = set(unavailable)
        cands: List[Candidate] = []
        for pid, reid_sim in self.vector_index.top_persons(
                vector, k=Config.MATCH_CANDIDATES, min_sim=Config.CANDIDATE_MIN_REID_SIM):
            part_sims = self.person_manager.part_similarities(pid, cues)
            fused, vetoed = fuse_scores(reid_sim, part_sims)
            if vetoed:
                continue
            c = Candidate(pid, fused, reid_sim, part_sims)
            if pid in unavailable:
                c.impossible = "seen next to this person in this camera"
            elif self.topology is not None and self.presence is not None and camera_id is not None:
                c.impossible = self.topology.impossible_reason(
                    self.presence.sightings(pid), camera_id, t,
                    positions=self.presence.positions(pid), foot=foot, now=now)
            cands.append(c)

        cands.sort(key=lambda c: -c.score)
        possible = [c for c in cands if c.impossible is None]
        top_any = cands[0] if cands else None
        blocked = top_any if (top_any is not None and top_any.impossible and top_any.score >= threshold) else None

        if not possible or possible[0].score < threshold:
            best = possible[0] if possible else None
            return MatchResult(pid=None, score=best.score if best else -1.0,
                               reid_sim=best.reid_sim if best else 0.0,
                               part_sims=best.part_sims if best else {},
                               best=best, blocked_by=blocked, candidates=cands)

        best = possible[0]
        rival = None
        if top_any is not best:                       # an impossible candidate scores higher
            rival = top_any
        elif len(possible) > 1 and possible[1].score > best.score - Config.AMBIGUITY_MARGIN:
            rival = possible[1]                       # a possible candidate is too close

        if rival is not None:
            return MatchResult(pid=None, score=best.score, reid_sim=best.reid_sim,
                               part_sims=best.part_sims, ambiguous=True, best=best, rival=rival,
                               blocked_by=blocked, candidates=cands)
        return MatchResult(pid=best.pid, score=best.score, reid_sim=best.reid_sim,
                           part_sims=best.part_sims, best=best, candidates=cands)

    @staticmethod
    def confirms(result: MatchResult, pid: int, min_score: float) -> bool:
        """
        Does appearance support giving `pid` to this person? Used to check a
        location-based guess (short-term re-acquisition): `pid` must look similar
        enough, must be possible, and no other possible person may look clearly
        more similar.
        """
        cand = next((c for c in result.candidates if c.pid == pid), None)
        if cand is None or cand.impossible is not None or cand.score < min_score:
            return False
        best_possible = max((c.score for c in result.candidates if c.impossible is None), default=-1.0)
        return cand.score >= best_possible - Config.AMBIGUITY_MARGIN
