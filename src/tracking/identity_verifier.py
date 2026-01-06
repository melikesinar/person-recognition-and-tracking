"""
Handles the logic for verifying and correcting person identities over time
to prevent rapid, incorrect ID switches.
"""
import time
import numpy as np
from datetime import datetime
from utils.config import Config

class IdentityVerifier:
    """
    Manages multi-stage identity verification to ensure robust ID corrections.
    """
    def __init__(self):
        """Initializes the IdentityVerifier."""
        self.verification_state = {} 
        self.last_redundancy_check = time.time()
        self.REDUNDANCY_CHECK_INTERVAL = 2.0 # Check every 2 seconds
        self.AUTO_MERGE_THRESHOLD = 0.85 # Very high similarity required to auto-merge
        print("Identity Verifier initialized.")

    def clear_state_for_track(self, track_id: int):
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

        for track_id, pid in list(buffer_manager.track_id_to_permanent_id_map.items()):
            # --- 0. Cooldown Check ---
            state = self.verification_state.get(track_id, {})
            last_switch = state.get("last_switch_time", 0)
            if (current_time - last_switch) < Config.ID_SWITCH_COOLDOWN:
                # Still in cooldown from last switch. Skip verification.
                continue

            buf = buffer_manager.active_track_buffers.get(track_id)
            if not buf or not buf["vectors"]:
                continue

            # Use average of recent vectors for stability
            avg_vector = np.mean(np.stack(buf["vectors"][-10:], axis=0), axis=0)
            avg_vector = avg_vector / np.linalg.norm(avg_vector)

            # 1. Calculate similarity to CURRENTLY assigned ID
            current_pid_sim = 0.0
            current_data = person_manager.known_persons_db.get(pid)
            if current_data and "vectors" in current_data:
                sims = [np.dot(v, avg_vector) for v in current_data["vectors"]]
                if sims:
                    current_pid_sim = max(sims)

            if current_pid_sim < Config.ID_DROP_THRESHOLD:
                state = self.verification_state.get(track_id, {})
                
                # Check if we are already tracking a "drop" for this track
                if state.get("candidate_pid") == -1: # -1 indicates DROP candidate
                    state["count"] += 1
                    state["last_seen"] = current_time
                else:
                    state = {
                        "candidate_pid": -1,
                        "count": 1,
                        "last_seen": current_time
                    }
                    self.verification_state[track_id] = state
                
                if state["count"] >= Config.ID_SWITCH_CONFIRMATION_THRESHOLD:
                    print(f"REVOCATION: T{track_id} dropped from P{pid} (Sim: {current_pid_sim:.2f} < {Config.ID_DROP_THRESHOLD}). Now Unknown.")
                    del buffer_manager.track_id_to_permanent_id_map[track_id]
                    self.clear_state_for_track(track_id)
                else:
                    self.verification_state[track_id] = state
                
                continue

            # 2. Find the best possible match in the database (Global Search)
            best_pid, best_score = faiss_manager.search(avg_vector, threshold=0.0)

            if best_pid is not None and best_pid != pid:
                # Condition to consider a switch: new match is significantly better than CURRENT match
                is_candidate = best_score > (current_pid_sim + Config.ID_SWITCH_THRESHOLD_MARGIN)

                if is_candidate:
                    # A potential switch is detected.
                    candidate_pid = state.get("candidate_pid")
                    
                    if candidate_pid == best_pid:
                        # This is a confirmation of the same candidate. Increment counter.
                        state["count"] = state.get("count", 0) + 1
                        state["last_seen"] = current_time
                        self.verification_state[track_id] = state # Ensure state is saved
                        
                        print(
                            f"CONFIRMATION: T{track_id} (P{pid}:{current_pid_sim:.2f}) -> P{best_pid}:{best_score:.2f}. Count: {state['count']}/{Config.ID_SWITCH_CONFIRMATION_THRESHOLD}")
                    else:
                        # This is a new candidate. Start a new verification process.
                        # Preserve last_switch_time
                        state["candidate_pid"] = best_pid
                        state["count"] = 1
                        state["last_seen"] = current_time
                        self.verification_state[track_id] = state
                        
                        print(f"CANDIDATE: T{track_id} (P{pid}:{current_pid_sim:.2f}) might be P{best_pid}:{best_score:.2f}. Starting count.")

                    # Check if the confirmation threshold is met
                    if state["count"] >= Config.ID_SWITCH_CONFIRMATION_THRESHOLD:
                        print(
                            f"SWITCH CONFIRMED: T{track_id} switched from P{pid} -> P{best_pid} after {state['count']} checks.")
                        buffer_manager.track_id_to_permanent_id_map[track_id] = best_pid
                        
                        # Apply Cooldown: Reset state but SET last_switch_time
                        state["candidate_pid"] = None
                        state["count"] = 0
                        state["last_switch_time"] = current_time
                        self.verification_state[track_id] = state

                elif state.get("candidate_pid"):
                    # The current frame does NOT confirm the candidate.
                    # Check if the counter should be reset due to timeout.
                    if (current_time - state.get("last_seen", 0)) > Config.ID_SWITCH_COUNTER_TTL_SECONDS:
                        # print(f"RESET: T{track_id} candidate P{state['candidate_pid']} expired. Resetting counter.")
                        state["candidate_pid"] = None
                        state["count"] = 0
                        self.verification_state[track_id] = state
            else:
                # No strong alternative match was found for this track.
                # If there was a pending verification, reset it due to timeout.
                if state.get("candidate_pid") and (current_time - state.get("last_seen", 0)) > Config.ID_SWITCH_COUNTER_TTL_SECONDS:
                    # print(f"RESET: T{track_id} candidate P{state['candidate_pid']} expired. Resetting counter.")
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
            
            # AGE CHECK A
            data_a = person_manager.known_persons_db[pid_a]
            age_a = (datetime.now() - data_a["created"]).total_seconds()
            if age_a > Config.MERGE_MAX_AGE_SECONDS:
                continue

            # Multi-Vector Retrieval
            vectors_a = data_a.get("vectors")
            if not vectors_a: continue

            for j in range(i + 1, len(active_pids)):
                pid_b = active_pids[j]
                if pid_b not in person_manager.known_persons_db: continue
                
                # AGE CHECK B
                data_b = person_manager.known_persons_db[pid_b]
                age_b = (datetime.now() - data_b["created"]).total_seconds()
                if age_b > Config.MERGE_MAX_AGE_SECONDS:
                    continue

                vectors_b = data_b.get("vectors")
                if not vectors_b: continue

                # --- Multi-Vector Similarity Check ---
                # Find max similarity between any vector in A and any vector in B
                # Optimization: We can break early if we exceed threshold, but finding max is safer for stats.
                pair_max_sim = -1.0
                for va in vectors_a:
                    for vb in vectors_b:
                        s = np.dot(va, vb)
                        if s > pair_max_sim:
                            pair_max_sim = s
                
                if pair_max_sim > self.AUTO_MERGE_THRESHOLD:
                    if pair_max_sim > best_sim:
                        best_sim = pair_max_sim
                        merge_pair = (pid_a, pid_b)

        # 3. Execute Merge if found
        if merge_pair:
            pid_keep, pid_drop = merge_pair
            if pid_keep > pid_drop:
                pid_keep, pid_drop = pid_drop, pid_keep
                
            print(f"AUTO-MERGE: Detected P{pid_keep} and P{pid_drop} are the same (Sim: {best_sim:.2f}). Merging...")
            
            # A. Merge in Database
            person_manager.merge_identities(keep_pid=pid_keep, drop_pid=pid_drop)
            
            # B. Update Live Tracks ACROSS ALL CAMERAS
            for node in camera_nodes:
                bm = node.buffer_manager
                for tid, pid in list(bm.track_id_to_permanent_id_map.items()):
                    if pid == pid_drop:
                        bm.track_id_to_permanent_id_map[tid] = pid_keep
                        print(f"  -> [Cam {node.camera_id}] Track T{tid} redirected from P{pid_drop} to P{pid_keep}")
