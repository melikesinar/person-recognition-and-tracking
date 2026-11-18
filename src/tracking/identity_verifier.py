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
from tracking.cross_camera_matcher import CrossCameraMatcher

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
        
        self.cross_camera_matcher = CrossCameraMatcher()

        # Pairs of PIDs (low, high) that were visible in the same camera at the same
        # time, i.e. proven to be different people. Never merged.
        self.cannot_link = set()

        logger.info("Identity Verifier initialized.")

    def clear_state_for_track(self, track_id: int):
        """Resets verification state for a specific track ID."""
        if track_id in self.verification_state:
            del self.verification_state[track_id]

    def verify_and_correct(self, buffer_manager, vector_index, current_match_threshold: float, person_manager):
        """
        Periodically checks identities and manages a multi-stage confirmation
        process before switching a person's ID.

        Args:
            buffer_manager (BufferManager): The main buffer manager instance.
            vector_index (VectorIndex): The shared vector search index.
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
            best_pid, best_score = vector_index.search(avg_vector, threshold=0.0)

            # Never switch to an identity another person holds in this camera right now
            held_here = set(buffer_manager.track_id_to_permanent_id_map.values())
            if best_pid is not None and best_pid != pid and best_pid not in held_here:
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
        Uses Multi-Vector comparison AND Spatial Consistency.
        """
        if not Config.LEGACY_AUTO_MERGE_ENABLED:
            return
        now = time.time()
        if (now - self.last_redundancy_check) < self.REDUNDANCY_CHECK_INTERVAL:
            return

        self.last_redundancy_check = now
        
        # 1. Collect ALL active PIDs from ALL cameras AND map them to their source nodes
        active_pids = []
        pid_to_node = {} # {pid: CameraNode} - stores the FIRST node seen (sufficient for pair check)

        for node in camera_nodes:
            pids_in_node = list(node.buffer_manager.track_id_to_permanent_id_map.values())
            for pid in pids_in_node:
                if pid not in pid_to_node:
                    pid_to_node[pid] = node
                    active_pids.append(pid)
        
        # If less than 2 people, no merge possible
        if len(active_pids) < 2:
            return

        # Two identities visible in the SAME camera at the same time are, physically,
        # two different people, however alike they look (e.g. same uniform). Such
        # pairs must never be merged.
        # The pairs are remembered ("cannot-link"), so they stay unmergeable later too.
        for node in camera_nodes:
            visible = sorted(set(node.buffer_manager.track_id_to_permanent_id_map.values()))
            for x in range(len(visible)):
                for y in range(x + 1, len(visible)):
                    self.cannot_link.add((visible[x], visible[y]))

        # Space-time rule (Faz 2): two identities seen in non-overlapping cameras too
        # close in time (closer than walking between them takes) are different people.
        resources = camera_nodes[0].resources if camera_nodes else {}
        topology, presence = resources.get('topology'), resources.get('presence')

        def seen_together(a, b):
            if (min(a, b), max(a, b)) in self.cannot_link:
                return True
            if topology is None or presence is None:
                return False
            if presence.proven_different(a, b):
                return True
            sa = presence.sightings(a)
            for cam_b, t_b in presence.sightings(b).items():
                if topology.impossible_reason(sa, cam_b, t_b) is not None:
                    self.cannot_link.add((min(a, b), max(a, b)))
                    return True
            return False

        # 2. Compare every pair
        merge_pair = None
        best_sim = -1.0
        reason = ""

        for i in range(len(active_pids)):
            pid_a = active_pids[i]
            if pid_a not in person_manager.known_persons_db: continue
            
            data_a = person_manager.known_persons_db[pid_a]
            # Age check...
            if "created" in data_a and (datetime.now() - data_a["created"]).total_seconds() > Config.MERGE_MAX_AGE_SECONDS:
                continue
            
            vectors_a = data_a.get("vectors")
            if not vectors_a: continue

            for j in range(i + 1, len(active_pids)):
                pid_b = active_pids[j]
                if pid_b not in person_manager.known_persons_db: continue
                if seen_together(pid_a, pid_b): continue
                
                data_b = person_manager.known_persons_db[pid_b]
                # Age check...
                if "created" in data_b and (datetime.now() - data_b["created"]).total_seconds() > Config.MERGE_MAX_AGE_SECONDS:
                    continue

                vectors_b = data_b.get("vectors")
                if not vectors_b: continue

                # --- A. Re-ID Similarity ---
                mat_a = np.array(vectors_a, dtype=np.float32) 
                mat_b = np.array(vectors_b, dtype=np.float32)
                sim_matrix = np.dot(mat_a, mat_b.T)
                visual_sim = np.max(sim_matrix)
                
                # --- B. Spatial Similarity ---
                spatial_score = 0.0
                node_a = pid_to_node[pid_a]
                node_b = pid_to_node[pid_b]
                
                # Only check spatial if cameras are different
                if node_a.camera_id != node_b.camera_id:
                    box_a = node_a.buffer_manager.last_known_box_for_pid.get(pid_a)
                    box_b = node_b.buffer_manager.last_known_box_for_pid.get(pid_b)
                    
                    if box_a and box_b:
                        spatial_score = self.cross_camera_matcher.calculate_spatial_score(
                            node_a.camera_id, box_a,
                            node_b.camera_id, box_b
                        )
                
                # --- C. Combined Score ---
                
                final_score = visual_sim
                current_reason = "Visual"
                
                # --- STRATEGY: Spatial Dominance ---
                # Use the powerful homography projection from CrossCameraMatcher.
                # If spatial_score is high (>0.8), it means the person is physically at the EXACT same location.
                
                if spatial_score > 0.80 and visual_sim > 0.60:
                    # STRONG SPATIAL MATCH
                    # Even if visual similarity is mediocre (e.g. 0.60 due to lighting/pose),
                    # we trust the physics (homography).
                    final_score = 0.96 # Force above AUTO_MERGE_THRESHOLD (0.85)
                    current_reason = f"Spatial-Dominant (Vis:{visual_sim:.2f}, Spat:{spatial_score:.2f})"
                
                elif spatial_score > 0.01:
                    # Standard Weighted Blend
                    # If visual is very low (<0.50), ignore spatial to prevent merging unrelated people 
                    # who just happen to overlap in 2D projection (occlusion).
                    if visual_sim < 0.50:
                        spatial_score = 0.0
                    
                    w = Config.SPATIAL_MERGE_SCORE_WEIGHT
                    final_score = (visual_sim * (1.0 - w)) + (spatial_score * w)
                    current_reason = f"Hybrid (Vis:{visual_sim:.2f}, Spat:{spatial_score:.2f})"

                if final_score > self.AUTO_MERGE_THRESHOLD:
                    if final_score > best_sim:
                        best_sim = final_score
                        merge_pair = (pid_a, pid_b)
                        reason = current_reason

        # 3. Execute Merge if found
        if merge_pair:
            pid_keep, pid_drop = merge_pair
            if pid_keep > pid_drop:
                pid_keep, pid_drop = pid_drop, pid_keep
                
            logger.info(f"AUTO-MERGE: P{pid_keep} & P{pid_drop} merged. Score: {best_sim:.2f} [{reason}]")
            
            # --- PRE-MERGE CONFLICT RESOLUTION ---
            # Scenario: We matched P3 (Cam A) with P4 (Cam B) spatially. So P4 is the REAL P3.
            # But what if Cam B already has a "P3" (Imposter/Error)?
            # If we just convert P4->P3, Cam B will have TWO P3s.
            # We must REVOKE the ID of the 'existing' P3 in any camera that also holds P4.
            
            for node in camera_nodes:
                bm = node.buffer_manager
                
                # Check if this camera holds the ID we are about to change (pid_drop)
                has_drop_id = False
                active_pids = list(bm.track_id_to_permanent_id_map.values())
                if pid_drop in active_pids:
                    has_drop_id = True
                
                if has_drop_id:
                    # If this camera holds pid_drop, it MUST NOT also hold pid_keep.
                    # If it does, that pid_keep is an imposter.
                    imposter_tracks = []
                    for tid, pid in bm.track_id_to_permanent_id_map.items():
                        if pid == pid_keep:
                            imposter_tracks.append(tid)
                    
                    for tid in imposter_tracks:
                        logger.warning(f"CONFLICT RESOLUTION: [Cam {node.camera_id}] Track T{tid} was labeled P{pid_keep}, but P{pid_drop} is verified as P{pid_keep}. Revoking T{tid} (Imposter).")
                        del bm.track_id_to_permanent_id_map[tid]
                        # Force re-verification logic to reset for this track
                        self.clear_state_for_track(tid)

            # A. Merge in Database
            person_manager.merge_identities(keep_pid=pid_keep, drop_pid=pid_drop)
            if presence is not None:
                presence.merge(pid_keep, pid_drop)
            # Whoever was proven different from pid_drop is now proven different from pid_keep
            for a, b in [p for p in self.cannot_link if pid_drop in p]:
                self.cannot_link.discard((a, b))
                other = b if a == pid_drop else a
                if other != pid_keep:
                    self.cannot_link.add((min(other, pid_keep), max(other, pid_keep)))
            
            # B. Update Live Tracks ACROSS ALL CAMERAS
            for node in camera_nodes:
                bm = node.buffer_manager
                for tid, pid in list(bm.track_id_to_permanent_id_map.items()):
                    if pid == pid_drop:
                        bm.track_id_to_permanent_id_map[tid] = pid_keep
                        if pid_drop in bm.last_known_box_for_pid:
                             bm.last_known_box_for_pid[pid_keep] = bm.last_known_box_for_pid.pop(pid_drop)
                        
                        logger.info(f"  -> [Cam {node.camera_id}] Track T{tid} redirected from P{pid_drop} to P{pid_keep}")
