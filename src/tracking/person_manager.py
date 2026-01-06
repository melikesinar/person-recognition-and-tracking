"""
Person enrollment and vector update management (Multi-Vector Gallery).
"""
import numpy as np
from datetime import datetime
from bson.binary import Binary
from utils.config import Config
from utils.math_utils import l2_normalize


class PersonManager:
    """
    Manages person enrollment, gallery vector updates, and ID assignment.
    Implements a 'Multi-Vector Gallery' strategy to handle varying poses/angles.
    """

    def __init__(self, known_persons_db, async_writer, faiss_manager):
        """
        Initialize person manager.

        Args:
            known_persons_db (dict): Reference to main person database
            async_writer (AsyncDBWriter): Database writer
            faiss_manager (FaissManager): Faiss index manager
        """
        self.known_persons_db = known_persons_db
        self.async_writer = async_writer
        self.faiss_manager = faiss_manager

        self.next_permanent_id = 1
        print("Person manager initialized (Multi-Vector Gallery Mode).")

    def set_next_id(self, next_id: int):
        """Set the next available permanent ID."""
        self.next_permanent_id = next_id

    def enroll_person(self, vectors: list) -> int:
        """
        Enrolls a new person with a unique Permanent ID.
        Initializes the gallery with a representative vector from the batch.

        Args:
            vectors (list): List of feature vectors for the person

        Returns:
            int: Newly assigned Permanent ID (P-ID)
        """
        if not vectors:
            raise ValueError("No vectors provided to enroll.")

        # Initialize gallery with the mean of the enrollment batch (Strong Seed)
        # Future improvement: Run K-Means here to start with 2-3 diverse vectors.
        initial_vector = l2_normalize(np.mean(np.stack(vectors, axis=0), axis=0))
        
        current_dt = datetime.now()
        new_pid = self.next_permanent_id

        # Gallery is a list of binary blobs for DB, list of arrays for Cache
        gallery_vectors = [initial_vector]

        # Create MongoDB document
        # We store 'vectors' as a list of Binaries
        new_person_doc = {
            "permanent_id": new_pid,
            "vectors": [Binary(v.tobytes()) for v in gallery_vectors],
            "created": current_dt,
            "last_seen": current_dt
        }

        # Queue async insert
        self.async_writer.queue_insert(new_person_doc)
        print(f"MongoDB INSERT: New person (P{new_pid}) queued. Gallery size: 1")

        # Add to local cache
        self.known_persons_db[new_pid] = {
            "vectors": gallery_vectors,
            "created": current_dt
        }

        print(f"CACHE ADD: P{new_pid} added to cache.")
        self.next_permanent_id += 1

        # Trigger index rebuild (Faiss will now index ALL vectors for this ID)
        self.faiss_manager.trigger_rebuild_if_needed(new_enrollment=True)

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
            # Just keep the ID alive.
            # print(f"UPDATE SKIPPED: P{perm_id} is occluded. No gallery update.")
            mongo_update_doc = {"$set": {"last_seen": datetime.now()}}
            self.async_writer.queue_update(perm_id, mongo_update_doc)
            return
        
        gallery = entry["vectors"] # List of np.ndarray
        
        # 1. Check similarity against ALL existing vectors in gallery
        # We want to know if this new vector is "redundant"
        similarities = [np.dot(v, new_vector) for v in gallery]
        max_sim = max(similarities)

        mongo_update_doc = {"$set": {"last_seen": datetime.now()}}

        if max_sim > Config.GALLERY_UPDATE_THRESHOLD:
            # We already have a vector very similar to this one.
            # No need to add it. Just update timestamp.
            # print(f"UPDATE SKIP: P{perm_id} new vector is redundant (Max Sim: {max_sim:.3f})")
            self.async_writer.queue_update(perm_id, mongo_update_doc)
            return

        # 2. It is a NEW/DIFFERENT pose (max_sim < Threshold). Add to gallery.
        # print(f"UPDATE DETECTED: P{perm_id} found new angle (Sim: {max_sim:.3f}). Adding to gallery.")
        
        if len(gallery) < Config.GALLERY_SIZE:
            # Case A: Gallery has space. Just append.
            gallery.append(new_vector)
        else:
            # Case B: Gallery is FULL.
            # Strategy: Find the pair of vectors in the gallery that are MOST similar to each other.
            # Merge them to free up 1 slot, then add the new vector.
            # This ensures we always keep the "most diverse" set of vectors.
            
            best_pair = (0, 1)
            highest_pair_sim = -1.0
            
            # Simple O(N^2) search (N=10, so it's tiny)
            for i in range(len(gallery)):
                for j in range(i + 1, len(gallery)):
                    sim = np.dot(gallery[i], gallery[j])
                    if sim > highest_pair_sim:
                        highest_pair_sim = sim
                        best_pair = (i, j)
            
            # Merge indices i and j
            idx1, idx2 = best_pair
            # print(f"GALLERY FULL: Merging similar vectors {idx1} and {idx2} (Sim: {highest_pair_sim:.3f})")
            
            merged_vec = l2_normalize(gallery[idx1] + gallery[idx2])
            
            # Remove the two old ones, add the merged one
            # Be careful with indices shifting when popping
            gallery.pop(max(idx1, idx2))
            gallery.pop(min(idx1, idx2))
            gallery.append(merged_vec)
            
            # Now we have space (N-1), add the NEW vector
            gallery.append(new_vector)

        # 3. Persist Changes
        # Convert entire gallery to Binary list for MongoDB
        # (Inefficient for huge lists, but for N=10 it's fine and safe)
        mongo_update_doc["$set"]["vectors"] = [Binary(v.tobytes()) for v in gallery]
        
        self.async_writer.queue_update(perm_id, mongo_update_doc)
        
        # Force a rebuild eventually because the index is now outdated
        # (We added a vector, so Faiss needs to know)
        # Optimization: We could just add to Faiss, but managing deletions is hard.
        # Rebuilding is safer for consistency.
        self.faiss_manager.trigger_rebuild_if_needed(new_enrollment=True)


    def merge_identities(self, keep_pid: int, drop_pid: int):
        """
        Merges two identities. Combines their galleries.
        """
        if keep_pid not in self.known_persons_db or drop_pid not in self.known_persons_db:
            return

        print(f"!!! MERGING IDENTITIES: P{drop_pid} -> INTO -> P{keep_pid} !!!")
        
        keep_data = self.known_persons_db[keep_pid]
        drop_data = self.known_persons_db[drop_pid]
        
        # Combine lists
        combined_gallery = keep_data["vectors"] + drop_data["vectors"]
        
        # Prune logic: If combined > GALLERY_SIZE, apply the "Merge Most Similar" logic repeatedly
        # until size is back to limit.
        while len(combined_gallery) > Config.GALLERY_SIZE:
            # Find most similar pair
            best_pair = (0, 1)
            highest_pair_sim = -1.0
            for i in range(len(combined_gallery)):
                for j in range(i + 1, len(combined_gallery)):
                    sim = np.dot(combined_gallery[i], combined_gallery[j])
                    if sim > highest_pair_sim:
                        highest_pair_sim = sim
                        best_pair = (i, j)
            
            idx1, idx2 = best_pair
            merged_vec = l2_normalize(combined_gallery[idx1] + combined_gallery[idx2])
            combined_gallery.pop(max(idx1, idx2))
            combined_gallery.pop(min(idx1, idx2))
            combined_gallery.append(merged_vec)
        
        # Update keep_pid
        keep_data["vectors"] = combined_gallery
        
        mongo_update_doc = {
            "$set": {
                "vectors": [Binary(v.tobytes()) for v in combined_gallery],
                "last_seen": datetime.now()
            }
        }
        self.async_writer.queue_update(keep_pid, mongo_update_doc)
        
        # Delete drop_pid
        self.delete_person(drop_pid)

    def delete_person(self, pid: int):
        """
        Deletes a person from database and cache.
        """
        self.async_writer.queue_delete(pid)
        if pid in self.known_persons_db:
            del self.known_persons_db[pid]
        self.async_writer.flush()
        self.faiss_manager.trigger_rebuild_if_needed(new_enrollment=True, force=True)

    def reset_all(self):
        """Resets entire database and cache."""
        self.async_writer.queue_reset()
        self.known_persons_db.clear()
        self.next_permanent_id = 1
        print("Person database reset queued.")