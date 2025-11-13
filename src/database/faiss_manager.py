"""
Faiss index management for fast similarity search.
Handles index building, loading, saving, and atomic swaps.
"""
import os
import time
import faiss
import threading
import numpy as np
from datetime import datetime
from bson.binary import Binary
from utils.config import Config


class FaissManager:
    """
    Manages the HNSW Faiss index for person Re-ID vector search.
    Implements dual-index pattern with atomic swap for non-blocking updates.
    """

    def __init__(self, mongo_collection):
        """
        Initialize Faiss manager.

        Args:
            mongo_collection: MongoDB collection for persisting index
        """
        self.faiss_collection = mongo_collection
        self.faiss_index = None
        self.index_to_pid_map = []
        self.faiss_lock = threading.Lock()

        # Rebuild tracking
        self.index_build_in_progress = False
        self.pending_enrollments = []
        self.last_index_rebuild = time.time()

        print("Faiss manager initialized.")

    def build_index(self, persons_db):
        """
        Builds a new HNSW Faiss index from person vectors.
        Uses atomic swap to minimize disruption to ongoing searches.

        Args:
            persons_db (dict): Dictionary of {pid: {"master_vector": np.ndarray, ...}}
        """
        print("Building new HNSW Faiss index (non-blocking)...")

        # 1. Collect vectors (without lock)
        all_vectors = []
        temp_pid_map = []

        db_snapshot = dict(persons_db)  # Shallow copy
        sorted_pids = sorted(db_snapshot.keys())

        for pid in sorted_pids:
            data = db_snapshot[pid]
            all_vectors.append(data['master_vector'])
            temp_pid_map.append(pid)

        # 2. Create new index (without lock)
        new_index = faiss.IndexHNSWFlat(
            Config.VECTOR_DIMENSION,
            Config.FAISS_M,
            faiss.METRIC_INNER_PRODUCT
        )
        new_index.hnsw.efConstruction = Config.FAISS_EF_CONSTRUCTION
        new_index.hnsw.efSearch = Config.FAISS_EF_SEARCH

        if not all_vectors:
            print("No vectors to index. HNSW index initialized empty.")
            with self.faiss_lock:
                self.faiss_index = new_index
                self.index_to_pid_map = []

            # Clean up MongoDB
            try:
                self.faiss_collection.delete_one({"name": Config.FAISS_INDEX_KEY})
            except Exception as e:
                print(f"!!!! FAISS (empty) index delete error !!!!: {e}")
            return

        # 3. Train and populate index (without lock)
        vector_matrix = np.stack(all_vectors, axis=0).astype('float32')
        new_index.train(vector_matrix)
        new_index.add(vector_matrix)

        print(f"New HNSW index built with {new_index.ntotal} vectors. Preparing atomic swap...")

        # 4. ATOMIC SWAP (critical section - very fast)
        with self.faiss_lock:
            old_index = self.faiss_index
            self.faiss_index = new_index
            self.index_to_pid_map = temp_pid_map
            print(f"✓ Index swapped atomically. Active vectors: {self.faiss_index.ntotal}")

        # 5. Persist to MongoDB (outside lock)
        self._save_index_to_db(new_index)

        self.index_build_in_progress = False

    def _save_index_to_db(self, index):
        """
        Saves Faiss index to MongoDB as binary data.

        Args:
            index: Faiss index object
        """
        try:
            faiss.write_index(index, Config.TEMP_FAISS_FILENAME)
            with open(Config.TEMP_FAISS_FILENAME, "rb") as f:
                index_buffer = f.read()

            self.faiss_collection.update_one(
                {"name": Config.FAISS_INDEX_KEY},
                {"$set": {
                    "index_data": Binary(index_buffer),
                    "last_updated": datetime.now()
                }},
                upsert=True
            )
            print(f"✓ HNSW Faiss index persisted to MongoDB.")
        except Exception as e:
            print(f"!!!! FAISS SAVE ERROR (non-critical) !!!!: {e}")
        finally:
            if os.path.exists(Config.TEMP_FAISS_FILENAME):
                os.remove(Config.TEMP_FAISS_FILENAME)

    def load_index_from_db(self, persons_db):
        """
        Loads Faiss index from MongoDB, or rebuilds if not found/stale.

        Args:
            persons_db (dict): Person database for validation
        """
        index_doc = self.faiss_collection.find_one({"name": Config.FAISS_INDEX_KEY})

        if index_doc and 'index_data' in index_doc:
            print("Loading Faiss index from MongoDB...")
            try:
                index_buffer = index_doc['index_data']
                with open(Config.TEMP_FAISS_FILENAME, "wb") as f:
                    f.write(index_buffer)

                loaded_index = faiss.read_index(Config.TEMP_FAISS_FILENAME)

                if hasattr(loaded_index, 'hnsw'):
                    loaded_index.hnsw.efSearch = Config.FAISS_EF_SEARCH

                # Validate index size
                if loaded_index.ntotal != len(persons_db):
                    print("Faiss index is stale (vector count mismatch). Rebuilding...")
                    raise ValueError("Index stale")

                with self.faiss_lock:
                    self.faiss_index = loaded_index

                    # Rebuild PID map
                    temp_pid_map = []
                    sorted_pids = sorted(persons_db.keys())
                    for pid in sorted_pids:
                        temp_pid_map.append(pid)
                    self.index_to_pid_map = temp_pid_map

                print("Faiss index loaded successfully from MongoDB.")
            except Exception as e:
                print(f"Failed to load Faiss index from MongoDB ({e}), rebuilding...")
                self.build_index(persons_db)
            finally:
                if os.path.exists(Config.TEMP_FAISS_FILENAME):
                    os.remove(Config.TEMP_FAISS_FILENAME)
        else:
            print("No Faiss index found in MongoDB. Building from scratch...")
            self.build_index(persons_db)

    def search(self, query_vector, threshold):
        """
        Performs thread-safe similarity search.

        Args:
            query_vector (np.ndarray): Query feature vector
            threshold (float): Minimum similarity threshold

        Returns:
            tuple: (matched_pid: int or None, best_similarity: float)
        """
        with self.faiss_lock:
            if self.faiss_index is None or self.faiss_index.ntotal == 0 or not self.index_to_pid_map:
                return None, -1.0

            try:
                query_vector_2d = query_vector.reshape(1, -1).astype('float32')
                D, I = self.faiss_index.search(query_vector_2d, k=1)
                best_similarity = D[0][0]

                if best_similarity >= threshold:
                    matched_index_position = I[0][0]
                    matched_pid = self.index_to_pid_map[matched_index_position]
                    return matched_pid, best_similarity
            except Exception as e:
                print(f"!!!! FAISS SEARCH ERROR !!!!: {e}")

        return None, -1.0

    def trigger_rebuild_if_needed(self, persons_db, new_enrollment=False):
        """
        Checks if index rebuild should be triggered based on batch size or time.

        Args:
            persons_db (dict): Current person database
            new_enrollment (bool): Whether this call is due to a new enrollment

        Returns:
            bool: True if rebuild was triggered
        """
        if new_enrollment:
            self.pending_enrollments.append(time.time())

        time_since_rebuild = time.time() - self.last_index_rebuild
        should_rebuild = (
                len(self.pending_enrollments) >= Config.BATCH_SIZE_TRIGGER or
                time_since_rebuild > Config.MIN_REBUILD_INTERVAL
        )

        if should_rebuild and not self.index_build_in_progress:
            print(f"Triggering index rebuild (pending: {len(self.pending_enrollments)})")
            self.index_build_in_progress = True
            self.pending_enrollments.clear()
            self.last_index_rebuild = time.time()

            # Start rebuild in background thread
            rebuild_thread = threading.Thread(
                target=self.build_index,
                args=(persons_db,),
                daemon=True
            )
            rebuild_thread.start()
            return True

        return False

    def get_index_stats(self):
        """
        Returns current index statistics.

        Returns:
            dict: Statistics (ntotal, build_in_progress, etc.)
        """
        with self.faiss_lock:
            return {
                "ntotal": self.faiss_index.ntotal if self.faiss_index else 0,
                "build_in_progress": self.index_build_in_progress,
                "pending_enrollments": len(self.pending_enrollments)
            }