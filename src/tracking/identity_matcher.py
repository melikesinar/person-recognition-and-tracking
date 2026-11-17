"""
Multi-cue identity matching (Faz 1).

Two-stage decision:
    1. Shortlist: OSNet similarity against every stored person (best vector of
       each person's gallery) gives the top MATCH_CANDIDATES people.
    2. Re-score: each candidate gets a fused score that also uses the per-part
       clothing colour similarity (see core/appearance_cues.fuse_scores). Parts
       that are not visible are simply left out. A candidate whose top AND bottom
       colours clearly differ is vetoed.

Keeping OSNet as the shortlist means colour alone can never pull in a stranger
who happens to wear the same colours; colour only re-orders and confirms people
OSNet already finds plausible.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from utils.config import Config
from core.appearance_cues import PartCues, fuse_scores


@dataclass
class MatchResult:
    pid: Optional[int]
    score: float
    reid_sim: float = 0.0
    part_sims: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v:.2f}" for k, v in self.part_sims.items()) or "no colour"
        return f"fused={self.score:.2f} (reid={self.reid_sim:.2f}; {parts})"


class IdentityMatcher:
    def __init__(self, person_manager, vector_index):
        self.person_manager = person_manager
        self.vector_index = vector_index

    def match(self, vector: np.ndarray, cues: List[PartCues], threshold: float) -> MatchResult:
        """
        Args:
            vector: OSNet embedding of the query (single frame or averaged).
            cues: colour cue observations of the query (one or more frames).
            threshold: minimum fused score to accept.
        Returns:
            MatchResult with pid=None if no candidate reaches the threshold.
            The best candidate is returned even if it is already active elsewhere;
            the caller decides how to treat that.
        """
        candidates = self.vector_index.top_persons(
            vector, k=Config.MATCH_CANDIDATES, min_sim=Config.CANDIDATE_MIN_REID_SIM
        )
        best = MatchResult(pid=None, score=-1.0)
        for pid, reid_sim in candidates:
            part_sims = self.person_manager.part_similarities(pid, cues)
            fused, vetoed = fuse_scores(reid_sim, part_sims)
            if vetoed:
                continue
            if fused > best.score:
                best = MatchResult(pid=pid, score=fused, reid_sim=reid_sim, part_sims=part_sims)

        if best.score < threshold:
            return MatchResult(pid=None, score=best.score, reid_sim=best.reid_sim, part_sims=best.part_sims)
        return best
