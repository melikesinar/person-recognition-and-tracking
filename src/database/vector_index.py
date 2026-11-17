"""
In-memory exact vector search over all people's gallery vectors.

Replaces the previous FaissManager (HNSW index rebuilt from MongoDB in a
background thread). Reasons:

* Consistency: the old index was rebuilt from MongoDB while writes to MongoDB
  were still queued asynchronously, so a just-enrolled person could be missing
  from search results for up to MIN_REBUILD_INTERVAL seconds and get enrolled
  a second time. Here, every gallery change is visible to the very next search.
* Scale: a site with a few hundred people x up to GALLERY_SIZE vectors is at most
  tens of thousands of 512-d vectors. One matrix product over that takes a few
  milliseconds on a CPU and is exact (HNSW is approximate).
* Simplicity: no rebuild threads, no index serialisation into the database.

If the gallery ever grows to millions of vectors, an approximate index (e.g.
Faiss) can be dropped in behind the same search() interface.
"""
import threading
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from utils.config import Config
from utils.logger import get_logger

logger = get_logger()


class VectorIndex:
    def __init__(self, k_neighbors: int = Config.SEARCH_K_NEIGHBORS):
        self.k_neighbors = k_neighbors
        self._lock = threading.Lock()
        self._galleries: Dict[int, np.ndarray] = {}
        # Flattened view, rebuilt lazily after any change
        self._matrix: Optional[np.ndarray] = None
        self._row_pids: Optional[np.ndarray] = None

    # --- Updates -------------------------------------------------------------

    def set_person(self, pid: int, vectors: Iterable[np.ndarray]):
        """Adds or replaces the gallery of one person."""
        vectors = list(vectors)
        with self._lock:
            if vectors:
                self._galleries[pid] = np.stack(vectors, axis=0).astype(np.float32)
            else:
                self._galleries.pop(pid, None)
            self._matrix = None

    def remove_person(self, pid: int):
        with self._lock:
            if self._galleries.pop(pid, None) is not None:
                self._matrix = None

    def clear(self):
        with self._lock:
            self._galleries.clear()
            self._matrix = None

    def load(self, persons: Dict[int, dict]):
        """Initialises the index from a {pid: {"vectors": [...]}} mapping."""
        self.clear()
        for pid, data in persons.items():
            self.set_person(pid, data.get("vectors") or [])
        logger.info(f"Vector index loaded: {self.size} vectors from {len(self._galleries)} identities.")

    # --- Queries -------------------------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            return sum(len(g) for g in self._galleries.values())

    def _snapshot(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if self._matrix is None and self._galleries:
                pids = list(self._galleries.keys())
                self._matrix = np.concatenate([self._galleries[p] for p in pids], axis=0)
                self._row_pids = np.concatenate(
                    [np.full(len(self._galleries[p]), p, dtype=np.int64) for p in pids]
                )
            return self._matrix, self._row_pids

    def top_persons(self, query_vector: np.ndarray, k: int, min_sim: float = -1.0) -> List[Tuple[int, float]]:
        """
        Returns up to k (pid, similarity) pairs, best first, where a person's
        similarity is the best match over all vectors in their gallery.
        Used to shortlist candidates that are then re-scored with other cues.
        """
        matrix, row_pids = self._snapshot()
        if matrix is None:
            return []
        sims = matrix @ query_vector.astype(np.float32).ravel()
        # Rows of one person are contiguous, so a segmented max gives per-person scores
        starts = np.flatnonzero(np.r_[True, row_pids[1:] != row_pids[:-1]])
        person_sims = np.maximum.reduceat(sims, starts)
        person_ids = row_pids[starts]
        order = np.argsort(-person_sims)[:k]
        return [(int(person_ids[i]), float(person_sims[i])) for i in order if person_sims[i] >= min_sim]

    def search(self, query_vector: np.ndarray, threshold: float) -> Tuple[Optional[int], float]:
        """
        k-NN voting search (same decision rule as the previous Faiss implementation):
        take the k most similar gallery vectors above `threshold`, the identity with
        the most votes wins, ties are broken by the best similarity.

        Returns:
            (pid, best_similarity) or (None, -1.0) if nothing passes the threshold.
        """
        matrix, row_pids = self._snapshot()
        if matrix is None:
            return None, -1.0

        sims = matrix @ query_vector.astype(np.float32).ravel()
        k = min(self.k_neighbors, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]

        candidates: Dict[int, list] = {}
        for idx in top:
            score = float(sims[idx])
            if score < threshold:
                continue
            pid = int(row_pids[idx])
            votes, best = candidates.get(pid, (0, -1.0))
            candidates[pid] = (votes + 1, max(best, score))

        if not candidates:
            return None, -1.0

        pid, (votes, best) = max(candidates.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        return pid, best
