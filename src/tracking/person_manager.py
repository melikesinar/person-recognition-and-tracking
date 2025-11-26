"""
Person enrollment and vector update management.
"""
import numpy as np
from datetime import datetime
from bson.binary import Binary
from utils.config import Config
from utils.math_utils import l2_normalize


class PersonManager:
    """
    Manages person enrollment, master vector updates, and ID assignment.
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

        print("Person manager initialized.")

    def set_next_id(self, next_id):
        """Set the next available permanent ID."""
        self.next_permanent_id = next_id



    def enroll_person(self, vectors):
        """
        Enrolls a new person with a unique Permanent ID.

        Args:
            vectors (list): List of feature vectors for the person

        Returns:
            int: Newly assigned Permanent ID (P-ID)
        """
        if not vectors:
            raise ValueError("No vectors provided to enroll.")

        # Calculate master vector (mean of all vectors)
        master = l2_normalize(np.mean(np.stack(vectors, axis=0), axis=0))
        current_dt = datetime.now()
        new_pid = self.next_permanent_id

        # Create MongoDB document
        new_person_doc = {
            "permanent_id": new_pid,
            "master_vector": Binary(master.tobytes()),
            "count": len(vectors),
            "created": current_dt,
            "last_seen": current_dt
        }

        # Queue async insert
        self.async_writer.queue_insert(new_person_doc)
        print(f"MongoDB INSERT: New person (P{new_pid}) queued for async write.")

        # Add to local cache
        self.known_persons_db[new_pid] = {
            "master_vector": master,
            "count": len(vectors),
            "created": current_dt
        }

        print(f"CACHE ADD: P{new_pid} added to cache.")
        self.next_permanent_id += 1

        # Trigger index rebuild if needed
        self.faiss_manager.trigger_rebuild_if_needed(new_enrollment=True)

        return new_pid

    def incremental_update(self, perm_id, new_vector):
        """
        Updates master vector using EMA (Exponential Moving Average).

        Args:
            perm_id (int): Permanent ID
            new_vector (np.ndarray): New feature vector
        """
        entry = self.known_persons_db.get(perm_id)
        if entry is None:
            return

        n = entry.get("count", 1)
        mongo_update_doc = {"$set": {"last_seen": datetime.now()}}

        # Update master vector if under limit
        if n < Config.MAX_VECTORS_FOR_UPDATE:
            master = entry["master_vector"]

            # EMA update
            updated_master = l2_normalize((1 - Config.EMA_ALPHA) * master + Config.EMA_ALPHA * new_vector)

            new_count = n + 1
            mongo_update_doc["$set"]["master_vector"] = Binary(updated_master.tobytes())
            mongo_update_doc["$set"]["count"] = new_count

            entry["master_vector"] = updated_master
            entry["count"] = new_count

            if new_count == Config.MAX_VECTORS_FOR_UPDATE:
                print(f"INFO: Person P{perm_id} vector locked after {new_count} updates.")

        # Queue async update
        self.async_writer.queue_update(perm_id, mongo_update_doc)

    def delete_person(self, pid):
        """
        Deletes a person from database and cache.

        Args:
            pid (int): Permanent ID to delete
        """
        # Queue async delete
        self.async_writer.queue_delete(pid)

        # Remove from cache
        if pid in self.known_persons_db:
            del self.known_persons_db[pid]
            print(f"DELETED: P{pid} removed from cache.")

        # Ensure the delete is processed by DB before rebuilding index
        self.async_writer.flush()

        # Trigger index rebuild to remove the person from search results
        self.faiss_manager.trigger_rebuild_if_needed(new_enrollment=True, force=True)


    def reset_all(self):
        """Resets entire database and cache."""
        self.async_writer.queue_reset()
        self.known_persons_db.clear()
        self.next_permanent_id = 1
        print("Person database reset queued.")