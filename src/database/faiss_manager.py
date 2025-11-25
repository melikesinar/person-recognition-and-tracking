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

    def __init__(self, mongo_handler):
        """
        Initialize Faiss manager.

        Args:
            mongo_handler (MongoHandler): Instance of the MongoDB handler.
        """
        self.mongo_handler = mongo_handler
        self.faiss_index = None
        self.index_to_pid_map = []
        self.faiss_lock = threading.Lock()

        # Rebuild tracking
        self.index_build_in_progress = False
        self.pending_enrollments = []
        self.last_index_rebuild = time.time()

        print("Faiss manager initialized.")

    def build_index(self):
        """
        Builds a new HNSW Faiss index directly from MongoDB vectors.
        Uses atomic swap to minimize disruption to ongoing searches.
        """
        print("Building new HNSW Faiss index from MongoDB (non-blocking)...")
        self.index_build_in_progress = True

        # 1. Fetch vectors and PIDs directly from MongoDB
        pids, vectors, _ = self.mongo_handler.load_vectors_for_faiss()

        # 2. Create new index (without lock)
        new_index = faiss.IndexHNSWFlat(
            Config.VECTOR_DIMENSION, Config.FAISS_M, faiss.METRIC_INNER_PRODUCT
        )
        new_index.hnsw.efConstruction = Config.FAISS_EF_CONSTRUCTION
        new_index.hnsw.efSearch = Config.FAISS_EF_SEARCH

        if not vectors:
            print("No vectors in DB. HNSW index initialized empty.")
            with self.faiss_lock:
                self.faiss_index = new_index
                self.index_to_pid_map = []
            self._save_index_to_db(new_index)  # Save the empty index
            self.index_build_in_progress = False
            return

        # 3. Train and populate index
        vector_matrix = np.stack(vectors, axis=0).astype('float32')
        new_index.train(vector_matrix)
        new_index.add(vector_matrix)

        print(f"New HNSW index built with {new_index.ntotal} vectors. Preparing atomic swap...")

        # 4. ATOMIC SWAP
        with self.faiss_lock:
            self.faiss_index = new_index
            self.index_to_pid_map = pids
            print(f"✓ Index swapped atomically. Active vectors: {self.faiss_index.ntotal}")

        # 5. Persist to MongoDB
        self._save_index_to_db(new_index)
        self.index_build_in_progress = False

    def _save_index_to_db(self, index):
        """
        Saves Faiss index to MongoDB as binary data.

        Args:
            index: Faiss index object
        """
        try:
            # We need to use a temporary file to write the index before reading as binary
            faiss.write_index(index, Config.TEMP_FAISS_FILENAME)
            with open(Config.TEMP_FAISS_FILENAME, "rb") as f:
                index_buffer = f.read()

            self.mongo_handler.faiss_collection.update_one(
                {"name": Config.FAISS_INDEX_KEY},
                {"$set": {
                    "index_data": Binary(index_buffer),
                    "pids": self.index_to_pid_map,  # Save the PID map along with the index
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

    def load_index_from_db(self):
        """
        Loads Faiss index and PID map from MongoDB, or rebuilds if not found.
        """
        index_doc = self.mongo_handler.faiss_collection.find_one({"name": Config.FAISS_INDEX_KEY})

        if index_doc and 'index_data' in index_doc and 'pids' in index_doc:
            print("Loading Faiss index from MongoDB...")
            try:
                index_buffer = index_doc['index_data']
                with open(Config.TEMP_FAISS_FILENAME, "wb") as f:
                    f.write(index_buffer)

                loaded_index = faiss.read_index(Config.TEMP_FAISS_FILENAME)
                if hasattr(loaded_index, 'hnsw'):
                    loaded_index.hnsw.efSearch = Config.FAISS_EF_SEARCH

                with self.faiss_lock:
                    self.faiss_index = loaded_index
                    self.index_to_pid_map = index_doc['pids']

                # Quick validation
                if self.faiss_index.ntotal != len(self.index_to_pid_map):
                    print("!!!! Index and PID map size mismatch, rebuilding... !!!!")
                    self.build_index()
                else:
                    print(f"Faiss index with {self.faiss_index.ntotal} vectors loaded successfully.")

            except Exception as e:
                print(f"Failed to load Faiss index from MongoDB ({e}), rebuilding...")
                self.build_index()
            finally:
                if os.path.exists(Config.TEMP_FAISS_FILENAME):
                    os.remove(Config.TEMP_FAISS_FILENAME)
        else:
            print("No Faiss index found in MongoDB. Building from scratch...")
            self.build_index()

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
            if self.faiss_index is None or self.faiss_index.ntotal == 0:
                return None, -1.0

            try:
                query_vector_2d = query_vector.reshape(1, -1).astype('float32')
                distances, indices = self.faiss_index.search(query_vector_2d, k=1)
                best_similarity = distances[0][0]

                if best_similarity >= threshold:
                    matched_index_pos = indices[0][0]
                    if matched_index_pos < len(self.index_to_pid_map):
                        matched_pid = self.index_to_pid_map[matched_index_pos]
                        return matched_pid, best_similarity
            except Exception as e:
                print(f"!!!! FAISS SEARCH ERROR !!!!: {e}")

        return None, -1.0

    def trigger_rebuild_if_needed(self, new_enrollment=False):
        """
        Checks if index rebuild should be triggered based on batch size or time.
        This is now self-sufficient and does not need persons_db.
        Args:
            new_enrollment (bool): Whether this call is due to a new enrollment
        """
        if new_enrollment:
            self.pending_enrollments.append(time.time())

        time_since_rebuild = time.time() - self.last_index_rebuild
        pending_count = len(self.pending_enrollments)
        
        # Rebuild if enough new people are pending, or if it's been a while since the last enrollment
        should_rebuild = (
            pending_count > 0 and (
                pending_count >= Config.BATCH_SIZE_TRIGGER or
                time_since_rebuild > Config.MIN_REBUILD_INTERVAL
            )
        )

        if should_rebuild and not self.index_build_in_progress:
            print(f"Triggering index rebuild (pending enrollments: {pending_count})")
            self.pending_enrollments.clear()
            self.last_index_rebuild = time.time()
            
            rebuild_thread = threading.Thread(target=self.build_index, daemon=True)
            rebuild_thread.start()

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