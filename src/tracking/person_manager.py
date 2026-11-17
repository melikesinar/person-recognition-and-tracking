"""
Person enrollment and vector update management (Multi-Vector Gallery).
"""
import numpy as np
from datetime import datetime
from bson.binary import Binary
from typing import Dict, Any, List, Optional, Tuple
from utils.config import Config
from utils.math_utils import l2_normalize
from utils.logger import get_logger
from core.appearance_cues import PART_NAMES, PartCues

logger = get_logger()

class PersonManager:
    """
    Manages person enrollment, gallery vector updates, and ID assignment.
    Implements a 'Multi-Vector Gallery' strategy to handle varying poses/angles.
    """

    def __init__(self, known_persons_db: Dict[int, Any], async_writer, vector_index):
        """
        Initialize person manager.

        Args:
            known_persons_db (dict): Reference to main person database
            async_writer (AsyncDBWriter | NullDBWriter): Database writer
            vector_index (VectorIndex): Search index, kept in sync with every gallery change
        """
        self.known_persons_db = known_persons_db
        self.async_writer = async_writer
        self.vector_index = vector_index

        self.next_permanent_id = 1
        logger.info("Person manager initialized (Multi-Vector Gallery Mode).")

    def set_next_id(self, next_id: int):
        """Set the next available permanent ID."""
        self.next_permanent_id = next_id

    def enroll_person(self, vectors: List[np.ndarray], cues: Optional[List[PartCues]] = None) -> int:
        """
        Enrolls a new person with a unique Permanent ID.
        Initializes the gallery with a representative vector from the batch.

        Args:
            vectors (list): List of feature vectors for the person
            cues (list): Part colour cues observed during enrollment (optional)

        Returns:
            int: Newly assigned Permanent ID (P-ID)
        """
        if not vectors:
            raise ValueError("No vectors provided to enroll.")

        # Initialize gallery with the mean of the enrollment batch (Strong Seed)
        # Using float32 for consistency
        mean_vector = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        initial_vector = l2_normalize(mean_vector)
        
        current_dt = datetime.now()
        new_pid = self.next_permanent_id

        # Gallery is a list of binary blobs for DB, list of arrays for Cache
        gallery_vectors = [initial_vector]

        # Create MongoDB document
        new_person_doc = {
            "permanent_id": new_pid,
            # Ensure vectors are bytes for BSON
            "vectors": [Binary(v.tobytes()) for v in gallery_vectors],
            "created": current_dt,
            "last_seen": current_dt
        }

        # Queue async insert
        self.async_writer.queue_insert(new_person_doc)
        logger.info(f"ENROLL: New person P{new_pid} created. Gallery size: 1")

        # Add to local cache
        self.known_persons_db[new_pid] = {
            "vectors": gallery_vectors,
            "cues": [],
            "created": current_dt
        }
        for c in cues or []:
            self.add_cues(new_pid, c)

        logger.info(f"CACHE ADD: P{new_pid} added to cache.")
        self.next_permanent_id += 1

        # Searchable immediately
        self.vector_index.set_person(new_pid, gallery_vectors)

        return new_pid

    def incremental_update(self, perm_id: int, new_vector: np.ndarray, is_occluded: bool = False):
        """
        Updates the person's gallery with a new vector if it represents a new angle/pose.
        Uses 'Merge Most Similar' strategy when gallery is full.
        
        Args:
            perm_id (int): Permanent ID
            new_vector (np.ndarray): New feature vector
            is_occluded (bool): If True, skips vector update to prevent pollution.
        """
        entry = self.known_persons_db.get(perm_id)
        if entry is None:
            return
        
        # 0. OCCLUSION CHECK (Pollution Guard)
        if is_occluded:
            # If the person is occluded (overlapped by another), we TRUST the tracking
            # but we DO NOT TRUST the visual features for learning.
            mongo_update_doc = {"$set": {"last_seen": datetime.now()}}
            self.async_writer.queue_update(perm_id, mongo_update_doc)
            return
        
        gallery = entry["vectors"] # List of np.ndarray
        
        # 1. Check similarity against ALL existing vectors in gallery
        # Optimized batch dot product
        # Stack gallery to matrix (N, 512) for faster calc
        gallery_matrix = np.stack(gallery, axis=0) 
        similarities = np.dot(gallery_matrix, new_vector)
        max_sim = np.max(similarities)

        mongo_update_doc = {"$set": {"last_seen": datetime.now()}}

        if max_sim > Config.GALLERY_UPDATE_THRESHOLD:
            # We already have a vector very similar to this one.
            self.async_writer.queue_update(perm_id, mongo_update_doc)
            return

        # 2. It is a NEW/DIFFERENT pose (max_sim < Threshold). Add to gallery.
        
        if len(gallery) < Config.GALLERY_SIZE:
            # Case A: Gallery has space. Just append.
            gallery.append(new_vector)
        else:
            # Case B: Gallery is FULL.
            # Strategy: Find the pair of vectors in the gallery that are MOST similar to each other.
            # Merge them to free up 1 slot.
            
            # Compute all pairwise similarities (N, N)
            sim_matrix = np.dot(gallery_matrix, gallery_matrix.T)
            # Mask diagonal and lower triangle to avoid self-match and duplicates
            np.fill_diagonal(sim_matrix, -1.0)
            
            # Find max index
            flat_idx = np.argmax(sim_matrix)
            idx1, idx2 = np.unravel_index(flat_idx, sim_matrix.shape)
            
            # If for some reason we picked the lower triangle (shouldn't happen with proper search but safe to normalize)
            # Actually unravelling gives row, col. Since matrix is symmetric, sim[i,j] == sim[j,i].
            
            # Merge indices idx1 and idx2
            merged_vec = l2_normalize(gallery[idx1] + gallery[idx2])
            
            # Remove the two old ones, add the merged one
            # Remove in descending order to avoid index shift issues
            pop_first = max(idx1, idx2)
            pop_second = min(idx1, idx2)
            
            gallery.pop(pop_first)
            gallery.pop(pop_second)
            gallery.append(merged_vec)
            
            # Now we have space (N-1), add the NEW vector
            gallery.append(new_vector)

        # 3. Persist Changes
        # Convert entire gallery to Binary list for MongoDB
        mongo_update_doc["$set"]["vectors"] = [Binary(v.tobytes()) for v in gallery]
        
        self.async_writer.queue_update(perm_id, mongo_update_doc)

        self.vector_index.set_person(perm_id, gallery)

    # ------------------------------------------------------------------ colour cue gallery

    def add_cues(self, pid: int, cues: Optional[PartCues]):
        """
        Stores one observation's part colour cues for a person (memory only).
        Kept as a rolling window of the most recent CUE_GALLERY_SIZE samples: clothing
        colour under different cameras/lighting is captured as the person moves.
        """
        entry = self.known_persons_db.get(pid)
        if entry is None or cues is None or not cues.available():
            return
        gallery = entry.setdefault("cues", [])
        gallery.append(cues)
        if len(gallery) > Config.CUE_GALLERY_SIZE:
            gallery.pop(0)

    def part_similarities(self, pid: int, query: List[PartCues]) -> Dict[str, float]:
        """
        Best colour similarity per body part between any of the query observations
        and any stored observation of the person. Parts never seen on both sides
        are absent from the result (and therefore ignored by the fusion).
        """
        entry = self.known_persons_db.get(pid)
        stored = entry.get("cues") if entry else None
        if not stored or not query:
            return {}
        best: Dict[str, float] = {}
        for name in PART_NAMES:
            q = [c.parts[name] for c in query if c.parts.get(name) is not None]
            g = [c.parts[name] for c in stored if c.parts.get(name) is not None]
            if q and g:
                best[name] = float(np.max(np.stack(q) @ np.stack(g).T))
        return best

    def merge_identities(self, keep_pid: int, drop_pid: int):
        """
        Merges two identities. Combines their galleries.
        """
        if keep_pid not in self.known_persons_db or drop_pid not in self.known_persons_db:
            return

        logger.warning(f"MERGING IDENTITIES: P{drop_pid} -> INTO -> P{keep_pid}")
        
        keep_data = self.known_persons_db[keep_pid]
        drop_data = self.known_persons_db[drop_pid]
        
        # Combine lists
        combined_gallery = keep_data["vectors"] + drop_data["vectors"]
        
        # Prune logic: If combined > GALLERY_SIZE
        while len(combined_gallery) > Config.GALLERY_SIZE:
            # Vectorized pair search
            gallery_matrix = np.stack(combined_gallery, axis=0)
            sim_matrix = np.dot(gallery_matrix, gallery_matrix.T)
            np.fill_diagonal(sim_matrix, -1.0) # Ignore self-similarity

            flat_idx = np.argmax(sim_matrix)
            idx1, idx2 = np.unravel_index(flat_idx, sim_matrix.shape)
            
            # Prevent merging identical indices if argmax fails (shouldn't happen if N > 1)
            if idx1 == idx2:
                 break

            merged_vec = l2_normalize(combined_gallery[idx1] + combined_gallery[idx2])
            
            pop_first = max(idx1, idx2)
            pop_second = min(idx1, idx2)
            
            combined_gallery.pop(pop_first)
            combined_gallery.pop(pop_second)
            combined_gallery.append(merged_vec)
        
        # Update keep_pid
        keep_data["vectors"] = combined_gallery
        keep_data["cues"] = (keep_data.get("cues", []) + drop_data.get("cues", []))[-Config.CUE_GALLERY_SIZE:]
        
        mongo_update_doc = {
            "$set": {
                "vectors": [Binary(v.tobytes()) for v in combined_gallery],
                "last_seen": datetime.now()
            }
        }
        self.async_writer.queue_update(keep_pid, mongo_update_doc)
        self.vector_index.set_person(keep_pid, combined_gallery)

        # Delete drop_pid
        self.delete_person(drop_pid)

    def delete_person(self, pid: int):
        """
        Deletes a person from database and cache.
        """
        self.async_writer.queue_delete(pid)
        if pid in self.known_persons_db:
            del self.known_persons_db[pid]
        self.vector_index.remove_person(pid)

    def reset_all(self):
        """Resets entire database and cache."""
        self.async_writer.queue_reset()
        self.known_persons_db.clear()
        self.vector_index.clear()
        self.next_permanent_id = 1
        logger.warning("Person database reset queued.")
