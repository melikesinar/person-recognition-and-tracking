"""
Buffer management and cleanup for tracking state.
"""
import time
from collections import defaultdict, deque
from utils.config import Config


class BufferManager:
    """
    Manages various buffers for tracking state and performs periodic cleanup.
    """

    def __init__(self):
        """Initialize buffer manager."""
        # Tracking buffers
        self.active_track_buffers = defaultdict(lambda: {
            "vectors": [],
            "crops": [],
            "quality_scores": [],
            "last_seen": time.time()
        })

        self.track_id_to_permanent_id_map = {}
        self.lost_pid_buffer = {}
        self.last_known_box_for_pid = {}

        # Update tracking
        self.last_update_time_for_pid = defaultdict(float)

        # Cleanup tracking
        self.last_buffer_cleanup = time.time()

        print("Buffer manager initialized.")

    def cleanup_stale_buffers(self, current_time, known_persons_db):
        """
        Periodic cleanup of stale buffers.

        Args:
            current_time (float): Current timestamp
            known_persons_db (dict): Reference to main person database
        """
        stale_pids = []

        # Check last_update_time_for_pid
        for pid in list(self.last_update_time_for_pid.keys()):
            if pid not in known_persons_db:
                if pid not in stale_pids:
                    stale_pids.append(pid)

        # Remove stale entries
        if stale_pids:
            for pid in stale_pids:
                if pid in self.last_update_time_for_pid:
                    del self.last_update_time_for_pid[pid]
                if pid in self.last_known_box_for_pid:
                    del self.last_known_box_for_pid[pid]

            print(f"BUFFER CLEANUP: Removed {len(stale_pids)} stale buffer(s).")

    def cleanup_stale_tracks(self, current_time):
        """
        Remove stale active track buffers (T-IDs).

        Args:
            current_time (float): Current timestamp
        """
        stale_tracks = [
            tid for tid, buf in self.active_track_buffers.items()
            if (current_time - buf.get("last_seen", current_time)) > 5.0
        ]

        for tid in stale_tracks:
            del self.active_track_buffers[tid]

    def cleanup_lost_pids(self, current_time):
        """
        Remove PIDs that have expired from short-term memory.

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

            if pid in self.last_known_box_for_pid:
                del self.last_known_box_for_pid[pid]

        return stale_lost_pids

    def add_to_lost_buffer(self, pid, box, current_time):
        """
        Add PID to lost buffer for re-acquisition.

        Args:
            pid (int): Permanent ID
            box (list): Last known bounding box
            current_time (float): Current timestamp
        """
        velocity = [0, 0]  # Placeholder for future Kalman filter

        self.lost_pid_buffer[pid] = {
            "disappear_time": current_time,
            "last_box": box,
            "velocity": velocity
        }

    def should_run_cleanup(self, current_time):
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

    def reset_all(self):
        """Reset all buffers."""
        self.active_track_buffers.clear()
        self.track_id_to_permanent_id_map.clear()
        self.lost_pid_buffer.clear()
        self.last_known_box_for_pid.clear()
        self.last_update_time_for_pid.clear()