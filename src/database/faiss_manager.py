"""
Faiss index management for fast similarity search.
Handles index building, loading, saving, and atomic swaps.
"""
import os
import time
import faiss
import tempfile
import threading
import numpy as np
from datetime import datetime
from bson.binary import Binary
from utils.config import Config
from utils.logger import get_logger

logger = get_logger()

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

        # Index settings
        self.ef_construction = Config.FAISS_EF_CONSTRUCTION
        self.ef_search = Config.FAISS_EF_SEARCH
        self.faiss_m = Config.FAISS_M

        logger.info("Faiss manager initialized.")

    def build_index(self):
        """
        Builds a new HNSW Faiss index directly from MongoDB vectors.
        Uses atomic swap to minimize disruption to ongoing searches.
        """
        logger.info("Building new HNSW Faiss index from MongoDB (non-blocking)...")
        self.index_build_in_progress = True

        try:
            # 1. Fetch vectors and PIDs directly from MongoDB
            pids, vectors, _ = self.mongo_handler.load_vectors_for_faiss()

            # 2. Create new index (without lock)
            new_index = faiss.IndexHNSWFlat(
                Config.VECTOR_DIMENSION, self.faiss_m, faiss.METRIC_INNER_PRODUCT
            )
            new_index.hnsw.efConstruction = self.ef_construction
            new_index.hnsw.efSearch = self.ef_search

            if not vectors:
                logger.info("No vectors in DB. HNSW index initialized empty.")
                with self.faiss_lock:
                    self.faiss_index = new_index
                    self.index_to_pid_map = []
                # Save the empty index to sync state
                self._save_index_to_db(new_index)
                return

            # 3. Train and populate index
            # Ensure float32 for Faiss
            vector_matrix = np.stack(vectors, axis=0).astype('float32')
            
            # Since HNSWFlat doesn't strictly require 'train' in the same way IVF does, 
            # calling it is safe but mainly for consistency if we switch index types later.
            new_index.train(vector_matrix)
            new_index.add(vector_matrix)

            logger.info(f"New HNSW index built with {new_index.ntotal} vectors. Preparing atomic swap...")

            # 4. ATOMIC SWAP
            with self.faiss_lock:
                self.faiss_index = new_index
                self.index_to_pid_map = pids
                logger.info(f"✓ Index swapped atomically. Active vectors: {self.faiss_index.ntotal}")

            # 5. Persist to MongoDB (in background potentially, but here synchronous is safer to prevent race)
            self._save_index_to_db(new_index)

        except Exception as e:
            logger.critical(f"CRITICAL ERROR during Faiss Rebuild: {e}")
        finally:
            self.index_build_in_progress = False

    def _save_index_to_db(self, index):
        """
        Saves Faiss index to MongoDB as binary data.

        Args:
            index: Faiss index object
        """
        temp_filename = None
        try:
            # Create a temporary file
            fd, temp_filename = tempfile.mkstemp()
            os.close(fd) # Close the file descriptor immediately, we'll let Faiss handle it

            faiss.write_index(index, temp_filename)
            
            with open(temp_filename, "rb") as f:
                index_buffer = f.read()

            self.mongo_handler.faiss_collection.update_one(
                {"name": Config.FAISS_INDEX_KEY},
                {"$set": {
                    "index_data": Binary(index_buffer),
                    "pids": self.index_to_pid_map,
                    "last_updated": datetime.now()
                }},
                upsert=True
            )
            logger.info("✓ HNSW Faiss index persisted to MongoDB.")
        except Exception as e:
            logger.error(f"FAISS SAVE ERROR (non-critical): {e}")
        finally:
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except OSError:
                    pass

    def load_index_from_db(self):
        """
        Loads Faiss index and PID map from MongoDB, or rebuilds if not found.
        """
        try:
            index_doc = self.mongo_handler.faiss_collection.find_one({"name": Config.FAISS_INDEX_KEY})

            if index_doc and 'index_data' in index_doc and 'pids' in index_doc:
                logger.info("Loading Faiss index from MongoDB...")
                temp_filename = None
                
                try:
                    index_buffer = index_doc['index_data']
                    
                    fd, temp_filename = tempfile.mkstemp()
                    os.write(fd, index_buffer)
                    os.close(fd)

                    loaded_index = faiss.read_index(temp_filename)
                    
                    # Restore runtime parameters that might not be saved in the file or need specific tuning
                    if hasattr(loaded_index, 'hnsw'):
                        loaded_index.hnsw.efSearch = self.ef_search

                    with self.faiss_lock:
                        self.faiss_index = loaded_index
                        self.index_to_pid_map = index_doc['pids']

                    # Quick validation
                    if self.faiss_index.ntotal != len(self.index_to_pid_map):
                        logger.warning("Index and PID map size mismatch, rebuilding...")
                        self.build_index()
                    else:
                        logger.info(f"Faiss index with {self.faiss_index.ntotal} vectors loaded successfully.")

                except Exception as e:
                    logger.warning(f"Failed to load Faiss index from MongoDB ({e}), rebuilding...")
                    self.build_index()
                finally:
                    if temp_filename and os.path.exists(temp_filename):
                        try:
                            os.remove(temp_filename)
                        except OSError:
                            pass
            else:
                logger.info("No Faiss index found in MongoDB. Building from scratch...")
                self.build_index()
        except Exception as e:
            logger.error(f"Error checking MongoDB for Faiss index: {e}")
            self.build_index()

    def search(self, query_vector, threshold):
        """
        Performs thread-safe similarity search using Voting (k-NN) strategy.
        Fetches Top-5 matches and votes for the most frequent PID.

        Args:
            query_vector (np.ndarray): Query feature vector
            threshold (float): Minimum similarity threshold

        Returns:
            tuple: (matched_pid: int or None, best_similarity: float)
        """
        # Minimal lock duration
        with self.faiss_lock:
            index_ref = self.faiss_index
            pid_map_ref = self.index_to_pid_map
            
        if index_ref is None or index_ref.ntotal == 0:
            return None, -1.0

        try:
            # Faiss search releases the GIL internally
            query_vector_2d = query_vector.reshape(1, -1).astype('float32')
            
            # Retrieve Top-5 candidates for robustness
            k_neighbors = 5
            distances, indices = index_ref.search(query_vector_2d, k=k_neighbors)
            
            candidates = {} # {pid: {'votes': 0, 'max_score': 0.0}}

            # Iterate through the k results
            for i in range(k_neighbors):
                score = distances[0][i]
                idx = indices[0][i]
                
                # Basic threshold filter
                if score < threshold:
                    continue
                
                # Map index to PID
                if 0 <= idx < len(pid_map_ref):
                    pid = pid_map_ref[idx]
                    
                    if pid not in candidates:
                        candidates[pid] = {'votes': 0, 'max_score': -1.0}
                    
                    candidates[pid]['votes'] += 1
                    if score > candidates[pid]['max_score']:
                        candidates[pid]['max_score'] = score

            if not candidates:
                return None, -1.0

            # Sort candidates: 
            # 1. By Votes (Descending) - Safety in numbers
            # 2. By Max Score (Descending) - Tie-breaker
            sorted_candidates = sorted(
                candidates.items(), 
                key=lambda item: (item[1]['votes'], item[1]['max_score']), 
                reverse=True
            )

            winner_pid, stats = sorted_candidates[0]
            
            # Optional: Require strictly > 1 vote if the gallery is large? 
            # For now, strict voting sort is sufficient.
            
            return winner_pid, stats['max_score']

        except Exception as e:
            logger.error(f"FAISS SEARCH ERROR: {e}")

        return None, -1.0

    def trigger_rebuild_if_needed(self, new_enrollment=False, force=False):
        """
        Checks if index rebuild should be triggered based on batch size or time.
        """
        if new_enrollment:
            self.pending_enrollments.append(time.time())

        # If a rebuild is already running, don't trigger another
        if self.index_build_in_progress:
            return

        time_since_rebuild = time.time() - self.last_index_rebuild
        pending_count = len(self.pending_enrollments)
        
        # Rebuild if:
        # 1. Forced
        # 2. We have pending enrollments AND (enough of them OR enough time passed)
        should_rebuild = force or (
            pending_count > 0 and (
                pending_count >= Config.BATCH_SIZE_TRIGGER or
                time_since_rebuild > Config.MIN_REBUILD_INTERVAL
            )
        )

        if should_rebuild:
            logger.debug(f"Triggering index rebuild (force={force}, pending: {pending_count})")
            self.pending_enrollments.clear()
            self.last_index_rebuild = time.time()
            
            rebuild_thread = threading.Thread(target=self.build_index, daemon=True)
            rebuild_thread.start()

    def get_index_stats(self):
        """
        Returns current index statistics.
        """
        with self.faiss_lock:
            return {
                "ntotal": self.faiss_index.ntotal if self.faiss_index else 0,
                "build_in_progress": self.index_build_in_progress,
                "pending_enrollments": len(self.pending_enrollments)
            }