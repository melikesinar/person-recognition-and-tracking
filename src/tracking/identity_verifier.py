"""
Handles the logic for verifying and correcting person identities over time
to prevent rapid, incorrect ID switches.
"""
import time
import numpy as np
from datetime import datetime
from utils.config import Config
from typing import Dict, Any, List, Optional, Tuple
from utils.logger import get_logger

logger = get_logger()

class IdentityVerifier:
    """
    Manages multi-stage identity verification to ensure robust ID corrections.
    """
    def __init__(self):
        """Initializes the IdentityVerifier."""
        # Stores state for ongoing verification processes per track
        # Key: track_id, Value: dict with keys 'candidate_pid', 'count', 'last_seen', 'last_switch_time'
        self.verification_state: Dict[int, Dict[str, Any]] = {} 
        
        self.last_redundancy_check = time.time()
        self.REDUNDANCY_CHECK_INTERVAL = 2.0 # Check every 2 seconds
        self.AUTO_MERGE_THRESHOLD = 0.85 # Very high similarity required to auto-merge
        
        logger.info("Identity Verifier initialized.")

    def clear_state_for_track(self, track_id: int):
        """Resets verification state for a specific track ID."""
        if track_id in self.verification_state:
            del self.verification_state[track_id]

    def verify_and_correct(self, buffer_manager, faiss_manager, current_match_threshold: float, person_manager):
        """
        Periodically checks identities and manages a multi-stage confirmation
        process before switching a person's ID.

        Args:
            buffer_manager (BufferManager): The main buffer manager instance.
            faiss_manager (FaissManager): The main Faiss manager instance.
            current_match_threshold (float): The current similarity threshold for matching.
            person_manager (PersonManager): The person manager to access current identity vectors.
        """
        current_time = time.time()

        # Iterate over a copy of items because we might modify the dictionary
        for track_id, pid in list(buffer_manager.track_id_to_permanent_id_map.items()):
            
            # --- 0. Cooldown & Lock Check ---
            # Respect explicit locks from BufferManager
            if buffer_manager.is_track_locked(track_id, current_time):
                continue

            state = self.verification_state.get(track_id, {})
            last_switch = state.get("last_switch_time", 0)
            if (current_time - last_switch) < Config.ID_SWITCH_COOLDOWN:
                continue

            buf = buffer_manager.active_track_buffers.get(track_id)
            if not buf or not buf["vectors"]:
                continue

            # Calculate robust average vector for the current track
            # Using the last N vectors helps filter out noise from bad frames
            recent_vectors = buf["vectors"][-10:]
            avg_vector = np.mean(np.stack(recent_vectors, axis=0), axis=0)
            avg_vector = avg_vector / (np.linalg.norm(avg_vector) + 1e-6) # Safer normalization

            # 1. Calculate similarity to CURRENTLY assigned ID
            current_pid_sim = 0.0
            current_data = person_manager.known_persons_db.get(pid)
            
            if current_data and "vectors" in current_data:
                # Compare against all vectors of the assigned person
                # We take the MAX similarity because the person might look different in different gallery shots
                person_vectors = current_data["vectors"]
                if person_vectors:
                    # Optimized dot product
                    sims = np.dot(person_vectors, avg_vector)
                    current_pid_sim = np.max(sims)

            # --- CHECK FOR REVOCATION (Dropping ID) ---
            if current_pid_sim < Config.ID_DROP_THRESHOLD:
                # If similarity drops too low, we might be tracking the wrong person (or no one known)
                self._handle_revocation(track_id, pid, current_pid_sim, current_time, buffer_manager)
                continue

            # --- CHECK FOR ID SWITCHING ---
            # 2. Find the best possible match in the database (Global Search)
            best_pid, best_score = faiss_manager.search(avg_vector, threshold=0.0)

            if best_pid is not None and best_pid != pid:
                # Condition to consider a switch: new match is significantly better than CURRENT match
                is_candidate = best_score > (current_pid_sim + Config.ID_SWITCH_THRESHOLD_MARGIN)

                if is_candidate:
                    self._handle_switch_candidate(
                        track_id, pid, current_pid_sim, best_pid, best_score, current_time, buffer_manager
                    )
                else:
                    self._check_timeout(track_id, current_time)
            else:
                self._check_timeout(track_id, current_time)

    def _handle_revocation(self, track_id, pid, current_sim, current_time, buffer_manager):
        """Internal logic to handle dropping an ID."""
        state = self.verification_state.get(track_id, {})
        
        # Check if we are already tracking a "drop" for this track (candidate_pid = -1)
        if state.get("candidate_pid") == -1: 
            state["count"] = state.get("count", 0) + 1
            state["last_seen"] = current_time
        else:
            state = {
                "candidate_pid": -1, # -1 indicates DROP candidate
                "count": 1,
                "last_seen": current_time,
                "last_switch_time": state.get("last_switch_time", 0) # Preserve cooldown info
            }
        
        self.verification_state[track_id] = state
        
        if state["count"] >= Config.ID_SWITCH_CONFIRMATION_THRESHOLD:
            logger.warning(f"REVOCATION: T{track_id} dropped from P{pid} (Sim: {current_sim:.2f} < {Config.ID_DROP_THRESHOLD}). Now Unknown.")
            del buffer_manager.track_id_to_permanent_id_map[track_id]
            # Reset state but keep the cooldown? Usually for revocation we just clear.
            self.clear_state_for_track(track_id)

    def _handle_switch_candidate(self, track_id, current_pid, current_sim, new_pid, new_score, current_time, buffer_manager):
        """Internal logic to handle switching to a new ID."""
        state = self.verification_state.get(track_id, {})
        candidate_pid = state.get("candidate_pid")
        
        if candidate_pid == new_pid:
            # Confirmation of same candidate
            state["count"] = state.get("count", 0) + 1
            state["last_seen"] = current_time
            logger.debug(f"CONFIRMATION: T{track_id} (P{current_pid}:{current_sim:.2f}) -> P{new_pid}:{new_score:.2f}. Count: {state['count']}/{Config.ID_SWITCH_CONFIRMATION_THRESHOLD}")
        else:
            # New candidate detected
            state["candidate_pid"] = new_pid
            state["count"] = 1
            state["last_seen"] = current_time
            logger.debug(f"CANDIDATE: T{track_id} (P{current_pid}:{current_sim:.2f}) might be P{new_pid}:{new_score:.2f}. Starting count.")

        self.verification_state[track_id] = state

        # Check if confirmed
        if state["count"] >= Config.ID_SWITCH_CONFIRMATION_THRESHOLD:
            logger.warning(f"SWITCH CONFIRMED: T{track_id} switched from P{current_pid} -> P{new_pid}.")
            buffer_manager.track_id_to_permanent_id_map[track_id] = new_pid
            
            # Apply Cooldown
            state = {
                "candidate_pid": None,
                "count": 0,
                "last_switch_time": current_time,
                "last_seen": current_time
            }
            self.verification_state[track_id] = state
            
            # Explicitly lock the track for a bit to prevent bouncing back
            buffer_manager.lock_track(track_id, duration=Config.ID_SWITCH_COOLDOWN)

    def _check_timeout(self, track_id, current_time):
        """Resets verification state if the candidate hasn't been seen recently."""
        state = self.verification_state.get(track_id)
        if state and state.get("candidate_pid") is not None:
             if (current_time - state.get("last_seen", 0)) > Config.ID_SWITCH_COUNTER_TTL_SECONDS:
                # Timeout expired, reset counter
                state["candidate_pid"] = None
                state["count"] = 0
                self.verification_state[track_id] = state

    def check_redundant_identities(self, person_manager, camera_nodes):
        """
        Scans ALL currently active PIDs across ALL cameras to see if any two represent the same person.
        Uses Multi-Vector comparison: If ANY vector of P1 matches ANY vector of P2 > Threshold, merge.
        """
        now = time.time()
        if (now - self.last_redundancy_check) < self.REDUNDANCY_CHECK_INTERVAL:
            return

        self.last_redundancy_check = now
        
        # 1. Collect ALL active PIDs from ALL cameras
        all_active_pids = set()
        for node in camera_nodes:
            all_active_pids.update(node.buffer_manager.track_id_to_permanent_id_map.values())
        
        active_pids = list(all_active_pids)
        
        # If less than 2 people, no merge possible
        if len(active_pids) < 2:
            return

        # 2. Compare every pair
        merge_pair = None
        best_sim = -1.0

        for i in range(len(active_pids)):
            pid_a = active_pids[i]
            if pid_a not in person_manager.known_persons_db: continue
            
            # Skip old identities to avoid processing huge history? 
            # Or skip if recently created? Config logic: MERGE_MAX_AGE_SECONDS checks if it's "fresh" enough to likely be a dupe.
            data_a = person_manager.known_persons_db[pid_a]
            if "created" in data_a:
                age_a = (datetime.now() - data_a["created"]).total_seconds()
                if age_a > Config.MERGE_MAX_AGE_SECONDS:
                    continue

            vectors_a = data_a.get("vectors")
            if not vectors_a: continue

            for j in range(i + 1, len(active_pids)):
                pid_b = active_pids[j]
                if pid_b not in person_manager.known_persons_db: continue
                
                data_b = person_manager.known_persons_db[pid_b]
                if "created" in data_b:
                    age_b = (datetime.now() - data_b["created"]).total_seconds()
                    if age_b > Config.MERGE_MAX_AGE_SECONDS:
                        continue

                vectors_b = data_b.get("vectors")
                if not vectors_b: continue

                # --- Multi-Vector Similarity Check (Vectorized) ---
                # Convert lists to numpy arrays for matrix multiplication
                # Ensure float32 for performance
                mat_a = np.array(vectors_a, dtype=np.float32) 
                mat_b = np.array(vectors_b, dtype=np.float32)
                
                # Compute all pairs similarity: (Na, Nb)
                sim_matrix = np.dot(mat_a, mat_b.T)
                
                # Find the maximum similarity in the matrix
                pair_max_sim = np.max(sim_matrix)
                
                if pair_max_sim > self.AUTO_MERGE_THRESHOLD:
                    if pair_max_sim > best_sim:
                        best_sim = pair_max_sim
                        merge_pair = (pid_a, pid_b)

        # 3. Execute Merge if found
        if merge_pair:
            pid_keep, pid_drop = merge_pair
            # Convention: keep the smaller ID (older one) usually, or merge into the more robust one.
            # Here keeping smaller ID for stability.
            if pid_keep > pid_drop:
                pid_keep, pid_drop = pid_drop, pid_keep
                
            logger.info(f"AUTO-MERGE: Detected P{pid_keep} and P{pid_drop} are the same (Sim: {best_sim:.2f}). Merging...")
            
            # A. Merge in Database
            person_manager.merge_identities(keep_pid=pid_keep, drop_pid=pid_drop)
            
            # B. Update Live Tracks ACROSS ALL CAMERAS
            for node in camera_nodes:
                bm = node.buffer_manager
                for tid, pid in list(bm.track_id_to_permanent_id_map.items()):
                    if pid == pid_drop:
                        bm.track_id_to_permanent_id_map[tid] = pid_keep
                        # Also update locks if any
                        if pid_drop in bm.last_known_box_for_pid:
                             bm.last_known_box_for_pid[pid_keep] = bm.last_known_box_for_pid.pop(pid_drop)
                        
                        logger.info(f"  -> [Cam {node.camera_id}] Track T{tid} redirected from P{pid_drop} to P{pid_keep}")
