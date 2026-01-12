"""
Buffer management and cleanup for tracking state.
"""
import time
from collections import defaultdict
from typing import Dict, List, Any
from utils.config import Config
from utils.logger import get_logger

logger = get_logger()

class BufferManager:
    """
    Manages various buffers for tracking state and performs periodic cleanup.
    """

    def __init__(self):
        """Initialize buffer manager."""
        # Tracking buffers: Store per-track vectors, crops, and metadata
        self.active_track_buffers: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "vectors": [],
            "crops": [],
            "quality_scores": [],
            "last_seen": time.time()
        })

        # Maps Track ID (T-ID) to Permanent ID (P-ID)
        self.track_id_to_permanent_id_map: Dict[int, int] = {}
        
        # Buffer for lost PIDs to facilitate re-acquisition (Short-term memory)
        self.lost_pid_buffer: Dict[int, Dict[str, Any]] = {}
        
        # Stores the last known spatial position for a PID
        self.last_known_box_for_pid: Dict[int, List[float]] = {}
        
        # Locks track IDs to prevent rapid switching of identities
        # Format: {tid: expiration_timestamp}
        self.locked_tracks: Dict[int, float] = {}

        # Last time a PID was updated (for long-term idle cleanup)
        self.last_update_time_for_pid: Dict[int, float] = defaultdict(float)

        # Cleanup timer
        self.last_buffer_cleanup = time.time()

        logger.info("Buffer manager initialized.")

    def cleanup_stale_buffers(self, current_time: float, known_persons_db: Dict[int, Any]):
        """
        Periodic cleanup of stale buffers.
        Removes long-idle entries from update trackers and position cache.

        Args:
            current_time (float): Current timestamp
            known_persons_db (dict): Reference to main person database
        """
        stale_pids = []

        # Check for idle PIDs that haven't been updated in a long time
        for pid, last_update_time in self.last_update_time_for_pid.items():
            if (current_time - last_update_time) > Config.BUFFER_MAX_IDLE_TIME:
                stale_pids.append(pid)

        # Remove stale entries from trackers
        if stale_pids:
            for pid in stale_pids:
                if pid in self.last_update_time_for_pid:
                    del self.last_update_time_for_pid[pid]
                if pid in self.last_known_box_for_pid:
                    del self.last_known_box_for_pid[pid]
            
            # Note: We do NOT remove them from 'known_persons_db' here.
            # That database is persistent. We only clean runtime caches.
            logger.debug(f"BUFFER CLEANUP: Removed {len(stale_pids)} stale buffer(s) due to inactivity.")

    def cleanup_stale_tracks(self, current_time: float):
        """
        Remove stale active track buffers (T-IDs).
        These are temporary tracks from the tracker (e.g., BoxMot) that are no longer present.

        Args:
            current_time (float): Current timestamp
        """
        # Identify stale tracks (not seen for > 5 seconds)
        # 5.0s is a hardcoded safety margin for temporary tracks
        stale_tracks = [
            tid for tid, buf in self.active_track_buffers.items()
            if (current_time - buf.get("last_seen", current_time)) > 5.0
        ]

        for tid in stale_tracks:
            del self.active_track_buffers[tid]
            
            # Also clean up the mapping if the track is gone
            if tid in self.track_id_to_permanent_id_map:
                del self.track_id_to_permanent_id_map[tid]
            
            if tid in self.locked_tracks:
                del self.locked_tracks[tid]

    def cleanup_lost_pids(self, current_time: float) -> List[int]:
        """
        Remove PIDs that have expired from short-term memory (Lost Buffer).
        If a person is lost for too long, they are removed from 'immediate re-acquisition' list.

        Args:
            current_time (float): Current timestamp

        Returns:
            list: List of expired PIDs
        """
        stale_lost_pids = [
            pid for pid, data in self.lost_pid_buffer.items()
            if (current_time - data["disappear_time"]) > Config.SHORT_TERM_MEMORY_TTL
        ]

        for pid in stale_lost_pids:
            del self.lost_pid_buffer[pid]
            # We don't necessarily delete from last_known_box_for_pid immediately here,
            # as that might be useful for longer-term spatial reasoning, 
            # but cleanup_stale_buffers will eventually catch it.

        return stale_lost_pids

    def add_to_lost_buffer(self, pid: int, box: List[float], current_time: float):
        """
        Add PID to lost buffer for re-acquisition.

        Args:
            pid (int): Permanent ID
            box (list): Last known bounding box
            current_time (float): Current timestamp
        """
        velocity = [0.0, 0.0]  # Placeholder for future Kalman filter implementation

        self.lost_pid_buffer[pid] = {
            "disappear_time": current_time,
            "last_box": box,
            "velocity": velocity
        }

    def should_run_cleanup(self, current_time: float) -> bool:
        """
        Check if periodic cleanup should run.

        Args:
            current_time (float): Current timestamp

        Returns:
            bool: True if cleanup should run
        """
        should_run = (current_time - self.last_buffer_cleanup) > Config.BUFFER_CLEANUP_INTERVAL
        if should_run:
            self.last_buffer_cleanup = current_time
        return should_run
    
    def is_track_locked(self, tid: int, current_time: float) -> bool:
        """
        Checks if a track ID is locked to a specific identity.
        """
        if tid in self.locked_tracks:
            if current_time < self.locked_tracks[tid]:
                return True
            else:
                del self.locked_tracks[tid] # Lock expired
                return False
        return False
        
    def lock_track(self, tid: int, duration: float = 2.0):
        """
        Locks a track ID for a specified duration.
        """
        self.locked_tracks[tid] = time.time() + duration

    def reset_all(self):
        """Reset all buffers."""
        self.active_track_buffers.clear()
        self.track_id_to_permanent_id_map.clear()
        self.lost_pid_buffer.clear()
        self.last_known_box_for_pid.clear()
        self.last_update_time_for_pid.clear()
        self.locked_tracks.clear()