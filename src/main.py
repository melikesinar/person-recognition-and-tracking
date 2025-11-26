# main_reid_vectors.py
import cv2
import time
import torch
import queue
import threading
import multiprocessing
import numpy as np
from pathlib import Path
from boxmot import BoostTrack
from datetime import datetime

# Import modular components
from utils.config import Config
from utils.visualizer import Visualizer
from core.quality_checker import QualityChecker
from core.detector import DetectorWorker
from core.reid_extractor import ReidExtractor
from database.faiss_manager import FaissManager
from database.async_writer import AsyncDBWriter
from database.mongo_handler import MongoHandler
from tracking.person_manager import PersonManager
from tracking.spatial_matcher import SpatialMatcher
from tracking.buffer_manager import BufferManager

# Validate configuration
Config.validate()
Config.print_config()

class PersonTrackerApp:
    """
    Main class that encapsulates all logic for the person Re-ID and tracking system,
    including thread management, database operations, and live processing.
    """

    def __init__(self):
        """
        Initializes the application with modular components.
        """
        print("System initializing...")

        # --- Core State ---
        self.known_persons_db = {}
        self.stop_event = threading.Event()

        # Live tuning thresholds
        self.current_match_threshold = Config.MATCH_SIMILARITY_THRESHOLD
        
        # Shared value for detection threshold (multiprocessing safe)
        self.detection_threshold_value = multiprocessing.Value('d', Config.DETECTION_SCORE_THRESHOLD)

        # TTA settings
        self.USE_TTA = Config.USE_TTA_DEFAULT
        self.TTA_HORIZONTAL_FLIP = Config.TTA_HORIZONTAL_FLIP

        # --- Database Connection ---
        try:
            self.mongo_handler = MongoHandler()
        except RuntimeError:
            exit()  # Exit if connection fails

        # --- Initialize Modular Components ---
        self.faiss_manager = FaissManager(self.mongo_handler)
        self.async_writer = AsyncDBWriter(
            self.mongo_handler.persons_collection,
            self.mongo_handler.faiss_collection,
            self.stop_event
        )
        self.person_manager = PersonManager(
            self.known_persons_db,
            self.async_writer,
            self.faiss_manager
        )
        self.spatial_matcher = SpatialMatcher()
        self.buffer_manager = BufferManager()
        self.quality_checker = QualityChecker()
        self.visualizer = Visualizer()

        # --- Thread/Process Communication (for detector) ---
        # Use multiprocessing Queues for inter-process communication
        self.frame_queue = multiprocessing.Queue(maxsize=1)
        self.detection_queue = multiprocessing.Queue(maxsize=1)

        # --- Load Models ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = 0 if self.device.type == 'cuda' else 'cpu'
        print(f"Using device: {self.device} (passed to worker as {device_str})")

        # --- Load Detector ---
        self.detector_worker_instance = DetectorWorker(
            self.frame_queue,
            self.detection_queue,
            self.detection_threshold_value,
            device_str
        )

        # --- Load ReidExtractor ---
        print("Loading Re-ID Model (OSNet)...")
        self.tracker = BoostTrack(
            reid_weights=Path(Config.REID_MODEL_PATH),
            device=(0 if self.device.type == 'cuda' else 'cpu'),
            half=False,
            with_reid=True,
            max_age=60,
            max_obs=100
        )

        try:
            self.reid_extractor = ReidExtractor(self.device, self.tracker)
        except RuntimeError as e:
            raise RuntimeError(e)

        # --- Load Data ---
        self.load_db()

    def advanced_quality_check(self, crop, bbox, frame_shape):
        """
        Wrapper for QualityChecker.assess_quality().
        Maintained for backward compatibility.
        """
        return self.quality_checker.assess_quality(crop, bbox, frame_shape)

    def calculate_sharpness(self, image):
        """
        Wrapper for QualityChecker.calculate_sharpness().
        Maintained for backward compatibility.
        """
        return self.quality_checker.calculate_sharpness(image)

    def load_db(self):
        """
        Initializes the system by loading the Faiss index and setting the next person ID.
        The known_persons_db now acts as a lazy-loaded cache and starts empty.
        """
        try:
            # 1. FaissManager now loads its index directly from MongoDB data
            self.faiss_manager.load_index_from_db()

            # 2. PersonManager gets the next available ID from MongoDB
            next_id = self.mongo_handler.get_next_id()
            self.person_manager.set_next_id(next_id)

            print("System ready. In-memory cache is empty and will be populated on-demand.")

        except Exception as e:
            print(f"!!!! Error during initialization process !!!!: {e}")
            # Ensure a fresh start if loading fails
            self.faiss_manager.build_index()
            self.known_persons_db.clear()
            self.person_manager.set_next_id(1)

    def get_box_center(self, box):
        """Wrapper for SpatialMatcher.get_box_center()."""
        return self.spatial_matcher.get_box_center(box)

    def calculate_iou(self, box1, box2):
        """Wrapper for SpatialMatcher.calculate_iou()."""
        return self.spatial_matcher.calculate_iou(box1, box2)

    def predict_next_position(self, last_box, time_elapsed):
        """Wrapper for SpatialMatcher.predict_next_position()."""
        return self.spatial_matcher.predict_next_position(last_box, time_elapsed)

    def adaptive_spatial_threshold(self, bbox):
        """Wrapper for SpatialMatcher.adaptive_spatial_threshold()."""
        return self.spatial_matcher.adaptive_spatial_threshold(bbox)

    def enroll_person(self, vectors):
        """Wrapper for PersonManager.enroll_person()."""
        return self.person_manager.enroll_person(vectors)

    def cleanup_stale_buffers(self, current_time):
        """Wrapper for BufferManager.cleanup_stale_buffers()."""
        return self.buffer_manager.cleanup_stale_buffers(current_time, self.known_persons_db)

    def incremental_update(self, perm_id, new_vector):
        """Wrapper for PersonManager.incremental_update()."""
        return self.person_manager.incremental_update(perm_id, new_vector)

    def find_best_match(self, query_vector):
        """Wrapper for FaissManager.search()."""
        return self.faiss_manager.search(query_vector, self.current_match_threshold)

    def _process_tracking_and_reid(self, detections, frame):
        """
        Handles BoostTrack update and batch Re-ID feature extraction.
        Implements QUALITY GATING: Only extracts features from high-quality crops.

        Args:
            detections (np.ndarray): Detections from the detector.
            frame (np.ndarray): The current video frame.

        Returns:
            tuple: (tracker_results_list, vector_map)
        """
        res = self.tracker.update(detections, frame)
        crops_to_process, track_ids_for_crops, tracker_results_list = [], [], []

        for *box, track_id, conf, cls, ind in res:
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            h, w, _ = frame.shape
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1: continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                # --- Quality Gating ---
                # Check quality BEFORE heavy ReID extraction
                is_good, _, _ = self.quality_checker.assess_quality(crop, box, frame.shape)
                
                if is_good:
                    crops_to_process.append(crop)
                    track_ids_for_crops.append(track_id)
                
                # We store the result regardless of quality so we can still draw the box
                tracker_results_list.append((box, track_id, conf, cls, ind, crop))

        all_vectors = []
        if crops_to_process:
            all_vectors = self.reid_extractor.extract_embeddings_from_crops(crops_to_process)

        vector_map = {}
        if len(all_vectors) == len(track_ids_for_crops):
            vector_map = {tid: vec for tid, vec in zip(track_ids_for_crops, all_vectors)}
        else:
            print(f"!!!! WARNING: Mismatch in batch Re-ID. Crops: {len(crops_to_process)}, Vectors: {len(all_vectors)}")

        return tracker_results_list, vector_map

    def _process_results(self, frame, tracker_results_list, vector_map):
        """
        Processes tracker results for matching, enrollment, and updates.
        Handles both High-Quality (ReID available) and Low-Quality (Spatial only) tracks.

        Args:
            frame (np.ndarray): The current video frame.
            tracker_results_list (list): List of tracker results.
            vector_map (dict): Map of track IDs to feature vectors (only for high-quality crops).

        Returns:
            set: A set of track IDs present in the current frame.
        """
        current_frame_track_ids = set()
        for (box, track_id, conf, cls, ind, crop) in tracker_results_list:
            current_frame_track_ids.add(track_id)
            current_vector = vector_map.get(track_id)
            
            x1, y1, x2, y2 = map(int, box)
            label, color = f"T{track_id}", (0, 165, 255)
            pid = self.buffer_manager.track_id_to_permanent_id_map.get(track_id)

            # --- High Quality: Re-ID Vector Available ---
            if current_vector is not None:
                if pid:
                    label, color = f"P{pid}", (0, 255, 0)
                    if (time.time() - self.buffer_manager.last_update_time_for_pid.get(pid, 0)) > Config.UPDATE_INTERVAL:
                        self.incremental_update(pid, current_vector)
                        self.buffer_manager.last_update_time_for_pid[pid] = time.time()
                else:
                    now = time.time()
                    spatial_match_pid, best_match_score = self.spatial_matcher.find_spatial_match(
                        box, self.buffer_manager.lost_pid_buffer, now
                    )
                    if spatial_match_pid is not None:
                        pid = spatial_match_pid
                        self.buffer_manager.track_id_to_permanent_id_map[track_id] = pid
                        label = f"P{pid} (ReAcq:{best_match_score:.2f})"
                        color = (0, 255, 128)
                        if pid in self.buffer_manager.lost_pid_buffer:
                            del self.buffer_manager.lost_pid_buffer[pid]
                    else:
                        match_id, match_sim = self.find_best_match(current_vector)
                        if match_id is not None:
                            pid = match_id
                            self.buffer_manager.track_id_to_permanent_id_map[track_id] = pid
                            label, color = f"P{pid} (S:{match_sim:.2f})", (0, 255, 0)
                        else:
                            # --- Enrollment Logic ---
                            # Note: Crop is already known to be "good" quality by _process_tracking_and_reid
                            # But we re-check 'reason' and 'q_score' if we want detailed logs/display
                            is_good, reason, q_score = self.advanced_quality_check(crop, box, frame.shape)
                            buf = self.buffer_manager.active_track_buffers[track_id]
                            
                            buf["vectors"].append(current_vector)
                            buf.setdefault("quality_scores", []).append(q_score)
                            avg_q = np.mean(buf["quality_scores"])
                            label = f"New? T{track_id} ({len(buf['vectors'])}/{Config.MIN_ENROLL_FRAMES})"
                            
                            buf["last_seen"] = time.time()
                            if len(buf["vectors"]) >= Config.MIN_ENROLL_FRAMES:
                                avg_q = np.mean(buf.get("quality_scores", [0.5]))
                                if avg_q >= 0.65:
                                    new_pid = self.enroll_person(buf["vectors"])
                                    if new_pid:
                                        self.buffer_manager.track_id_to_permanent_id_map[track_id] = new_pid
                                        if track_id in self.buffer_manager.active_track_buffers:
                                            del self.buffer_manager.active_track_buffers[track_id]
                                else:
                                    buf["vectors"].clear(); buf.get("quality_scores", []).clear()

            # --- Low Quality: Spatial Tracking Only ---
            else:
                if pid:
                    # We know who this is, but the image is currently bad
                    label, color = f"P{pid} (LowQ)", (100, 255, 100) # Slightly dimmer green
                else:
                    # Unknown person and bad quality -> Try spatial recovery only
                    now = time.time()
                    spatial_match_pid, best_match_score = self.spatial_matcher.find_spatial_match(
                        box, self.buffer_manager.lost_pid_buffer, now
                    )
                    if spatial_match_pid is not None:
                        pid = spatial_match_pid
                        self.buffer_manager.track_id_to_permanent_id_map[track_id] = pid
                        label = f"P{pid} (LowQ-ReAcq)"
                        color = (100, 255, 128)
                        if pid in self.buffer_manager.lost_pid_buffer:
                            del self.buffer_manager.lost_pid_buffer[pid]
                    else:
                        label, color = f"T{track_id} (LowQ)", (0, 100, 155)

            # --- Common Update Logic ---
            if pid is not None:
                self.buffer_manager.last_known_box_for_pid[pid] = box
                if pid not in self.known_persons_db:
                    person_doc = self.mongo_handler.get_person_by_id(pid)
                    if person_doc: self.known_persons_db[pid] = person_doc
                
                if pid in self.known_persons_db:
                    self.known_persons_db[pid]['last_seen_timestamp'] = time.time()

            self.visualizer.draw_person_bbox(frame, (x1, y1, x2, y2), label, color)
        return current_frame_track_ids

    def _perform_cleanup(self, current_frame_track_ids):
        """
        Performs periodic cleanup of stale tracks and buffers.

        Args:
            current_frame_track_ids (set): Set of track IDs in the current frame.
        """
        now = time.time()
        self.buffer_manager.cleanup_stale_tracks(now)
        disappeared_track_ids = [
            tid for tid in self.buffer_manager.track_id_to_permanent_id_map
            if tid not in current_frame_track_ids
        ]
        for tid in disappeared_track_ids:
            pid = self.buffer_manager.track_id_to_permanent_id_map.pop(tid, None)
            if pid and pid in self.buffer_manager.last_known_box_for_pid:
                self.buffer_manager.add_to_lost_buffer(
                    pid, self.buffer_manager.last_known_box_for_pid[pid], now
                )
        # Clean up persons from the active cache if they haven't been seen for a while
        # This is optional but good for memory management over very long runs
        # Example: remove if not seen for 10 minutes
        pids_to_evict = [
            pid for pid, data in self.known_persons_db.items()
            if now - data.get('last_seen_timestamp', 0) > 600
        ]
        for pid in pids_to_evict:
            del self.known_persons_db[pid]
            print(f"Evicted P{pid} from active cache due to inactivity.")

        self.buffer_manager.cleanup_lost_pids(now)
        if self.buffer_manager.should_run_cleanup(now):
            self.cleanup_stale_buffers(now)

    def run(self):
        """
        The main application loop. It handles video capture, tracking update,
        batch feature extraction, Re-ID matching, spatial re-acquisition,
        enrollment logic, drawing results, and user input processing.
        """
        self.detector_worker_instance.start()

        # cap = cv2.VideoCapture("../videos/3.mp4")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("!!!! ERROR: Could not open camera. !!!!")
            return

        with torch.inference_mode():
            detections = np.empty((0, 6), dtype=np.float32)
            prev_time = 0
            frame_count = 0

            while True:
                success, frame = cap.read()
                if not success:
                    print("Video stream ended.")
                    break

                frame_count += 1
                current_time = time.time()
                time_diff = current_time - prev_time
                fps = 1.0 / time_diff if time_diff > 0.0001 else 0.0
                prev_time = current_time

                if frame_count % Config.DETECTION_INTERVAL == 0:
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass
                try:
                    detections = self.detection_queue.get_nowait()
                except queue.Empty:
                    pass

                tracker_results_list, vector_map = self._process_tracking_and_reid(detections, frame)
                current_frame_track_ids = self._process_results(frame, tracker_results_list, vector_map)
                self._perform_cleanup(current_frame_track_ids)

                self.visualizer.draw_hud(
                    frame, fps, self.detection_threshold_value.value,
                    self.current_match_threshold, len(self.known_persons_db), self.USE_TTA
                )
                cv2.imshow("ReID Vectors (MongoDB - Faiss)", frame)
                if self.handle_keyboard_input() == 'q':
                    break

    def handle_keyboard_input(self):
        """
        Processes user keyboard input for control actions, including quitting,
        live-tuning thresholds, resetting the database, and debugging status.

        Returns:
            str | int: The key pressed, or 'q' for quit signal.
        """
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            self.detector_worker_instance.stop()  # Signal detector thread to stop
            return 'q'

        # Live-tuning and other controls...
        elif key == ord('u'):
            new_val = min(1.0, self.detection_threshold_value.value + 0.05)
            self.detector_worker_instance.set_detection_threshold(new_val)
        elif key == ord('j'):
            new_val = max(0.0, self.detection_threshold_value.value - 0.05)
            self.detector_worker_instance.set_detection_threshold(new_val)
        elif key == ord('y'):
            self.current_match_threshold = min(1.0, self.current_match_threshold + 0.05)
        elif key == ord('h'):
            self.current_match_threshold = max(0.0, self.current_match_threshold - 0.05)
        elif key == ord('t'):
            self.USE_TTA = not self.USE_TTA
            self.reid_extractor.set_tta_status(self.USE_TTA)

        elif key == ord('r'):
            print("RESET: Queuing MongoDB clear operation...")
            self.person_manager.reset_all()
            self.buffer_manager.reset_all()
            self.faiss_manager.build_index()  # No longer needs persons_db
            print("Program memory (cache and index) has been cleared.")

        elif key == ord('d'):
            # Debugging status...
            print("\n--- DB STATUS (d) ---")
            print(f"Persons in ACTIVE CACHE: {len(self.known_persons_db)}")
            stats = self.faiss_manager.get_index_stats()
            print(f"Vectors in Faiss index: {stats['ntotal']}")
            print(f"Next ID: {self.person_manager.next_permanent_id}")
            print("------------------\n")

        elif key == ord('k'):
            if self.known_persons_db:
                try:
                    last_id = max(self.known_persons_db.keys())
                    self.person_manager.delete_person(last_id)
                    # Clear from cache
                    if last_id in self.known_persons_db:
                        del self.known_persons_db[last_id]
                    # Clear associated buffers
                    self.buffer_manager.active_track_buffers.clear()
                    self.buffer_manager.track_id_to_permanent_id_map.clear()
                    self.buffer_manager.lost_pid_buffer.clear()
                    print(f"DELETED (k): P{last_id} delete command issued.")
                except Exception as e:
                    print(f"Error deleting person: {e}")
            else:
                print("DB is already empty. Nothing to delete.")

        return key
    
    def cleanup(self):
        """Releases all resources gracefully."""
        self.stop_event.set()
        print("Cleaning up and exiting...")
        self.detector_worker_instance.stop()
        self.async_writer.shutdown()
        cv2.destroyAllWindows()
        self.mongo_handler.close()
        print(f"Exited cleanly. DB entries in active cache: {len(self.known_persons_db)}")

# --- Start the Application ---
if __name__ == "__main__":
    app = None
    try:
        app = PersonTrackerApp()
        app.run()
    except RuntimeError as e:
        print(f"!!!! CRITICAL RUNTIME ERROR !!!!: {e}")
    except Exception as e:
        print(f"!!!! UNEXPECTED ERROR OCCURRED !!!!: {e}")
    finally:
        if app:
            app.cleanup()