"""
Handles the logic for verifying and correcting person identities over time
to prevent rapid, incorrect ID switches.
"""
import time
import numpy as np
from utils.config import Config

class IdentityVerifier:
    """
    Manages multi-stage identity verification to ensure robust ID corrections.
    """
    def __init__(self):
        """Initializes the IdentityVerifier."""
        self.verification_state = {}  # {track_id: {"candidate_pid": pid, "count": n, "last_seen": t}}
        print("Identity Verifier initialized.")

    def clear_state_for_track(self, track_id: int):
        """
        Removes the verification state for a track that has disappeared.

        Args:
            track_id (int): The track ID to clear.
        """
        if track_id in self.verification_state:
            del self.verification_state[track_id]

    def verify_and_correct(self, buffer_manager, faiss_manager, current_match_threshold: float):
        """
        Periodically checks identities and manages a multi-stage confirmation
        process before switching a person's ID.

        Args:
            buffer_manager (BufferManager): The main buffer manager instance.
            faiss_manager (FaissManager): The main Faiss manager instance.
            current_match_threshold (float): The current similarity threshold for matching.
        """
        current_time = time.time()

        for track_id, pid in list(buffer_manager.track_id_to_permanent_id_map.items()):
            # 1. COOLDOWN CHECK
            # If this track recently switched, do not touch it.
            lock_expiry = buffer_manager.locked_tracks.get(track_id, 0)
            if current_time < lock_expiry:
                continue

            buf = buffer_manager.active_track_buffers.get(track_id)
            if not buf or not buf["vectors"]:
                continue

            # Use average of recent vectors for stability
            avg_vector = np.mean(np.stack(buf["vectors"][-10:], axis=0), axis=0)
            avg_vector = avg_vector / np.linalg.norm(avg_vector)

            # Find the best possible match in the database
            best_pid, best_score = faiss_manager.search(avg_vector, threshold=0.0)

            if best_pid is not None and best_pid != pid:
                # Condition to consider a switch: new match is significantly better
                is_candidate = best_score > (current_match_threshold + Config.ID_SWITCH_THRESHOLD_MARGIN)
                
                state = self.verification_state.get(track_id)

                if is_candidate:
                    # A potential switch is detected.
                    if state and state["candidate_pid"] == best_pid:
                        # This is a confirmation of the same candidate. Increment counter.
                        state["count"] += 1
                        state["last_seen"] = current_time
                        print(f"CONFIRMATION: T{track_id} (P{pid}) still looks like P{best_pid}. Count: {state['count']}/{Config.ID_SWITCH_CONFIRMATION_THRESHOLD}")
                    else:
                        # This is a new candidate. Start a new verification process.
                        self.verification_state[track_id] = {
                            "candidate_pid": best_pid,
                            "count": 1,
                            "last_seen": current_time
                        }
                        print(f"CANDIDATE: T{track_id} (P{pid}) might be P{best_pid}. Starting count.")

                    # Check if the confirmation threshold is met
                    if self.verification_state[track_id]["count"] >= Config.ID_SWITCH_CONFIRMATION_THRESHOLD:
                        print(f"SWITCH CONFIRMED: T{track_id} switched from P{pid} -> P{best_pid}. Locking for {Config.ID_SWITCH_COOLDOWN}s.")
                        
                        # Apply Switch
                        buffer_manager.track_id_to_permanent_id_map[track_id] = best_pid
                        
                        # Apply Lock
                        buffer_manager.locked_tracks[track_id] = current_time + Config.ID_SWITCH_COOLDOWN
                        
                        self.clear_state_for_track(track_id) # Clean up state after switch
                
                elif state:
                    # The current frame does NOT confirm the candidate.
                    # Check if the counter should be reset due to timeout.
                    if (current_time - state["last_seen"]) > Config.ID_SWITCH_COUNTER_TTL_SECONDS:
                        print(f"RESET: T{track_id} candidate P{state['candidate_pid']} expired. Resetting counter.")
                        self.clear_state_for_track(track_id)
            else:
                # No strong alternative match was found for this track.
                # If there was a pending verification, reset it due to timeout.
                state = self.verification_state.get(track_id)
                if state and (current_time - state["last_seen"]) > Config.ID_SWITCH_COUNTER_TTL_SECONDS:
                    print(f"RESET: T{track_id} candidate P{state['candidate_pid']} expired. Resetting counter.")
                    self.clear_state_for_track(track_id)