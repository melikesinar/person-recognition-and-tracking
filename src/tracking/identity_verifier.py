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

    def verify_and_correct(self, buffer_manager, faiss_manager, current_match_threshold: float):
        """Standard track-based verification."""
        # (Legacy logic preserved if needed, but mainly we focus on redundancy check now)
        pass 

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
