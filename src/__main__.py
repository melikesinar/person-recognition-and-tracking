# main_reid_vectors.py
import cv2
import time
import queue
import torch
import threading
import numpy as np
import multiprocessing
from pathlib import Path
from boxmot import BoostTrack

# Import modular components
from utils.config import Config
from utils.visualizer import Visualizer
from core.detector import DetectorWorker
from core.reid_extractor import ReidExtractor
from core.quality_checker import QualityChecker
from database.faiss_manager import FaissManager
from database.async_writer import AsyncDBWriter
from database.mongo_handler import MongoHandler
from tracking.person_manager import PersonManager
from tracking.buffer_manager import BufferManager
from tracking.spatial_matcher import SpatialMatcher
from tracking.identity_verifier import IdentityVerifier

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
        except RuntimeError as e:
            raise RuntimeError(f"Failed to initialize MongoDB handler: {e}")

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
        # Shared buffer manager (Global IDs)
        self.buffer_manager = BufferManager()
        self.quality_checker = QualityChecker()
        self.visualizer = Visualizer()
        self.identity_verifier = IdentityVerifier()
        self.spatial_matcher = SpatialMatcher()

        # --- Thread/Process Communication (for detector) ---
        # Use multiprocessing Queues for inter-process communication
        self.frame_queue = multiprocessing.Queue(maxsize=2) # Slightly larger for multi-cam
        self.detection_queue = multiprocessing.Queue(maxsize=2)
        self.detector_ready_event = multiprocessing.Event()

        # --- Load Models ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = 0 if self.device.type == 'cuda' else 'cpu'
        print(f"Using device: {self.device} (passed to worker as {device_str})")

        # --- Load Detector ---
        self.detector_worker_instance = DetectorWorker(
            self.frame_queue,
            self.detection_queue,
            self.detection_threshold_value,
            device_str,
            self.detector_ready_event
        )

        # --- Load Trackers (Per Camera) ---
        # Determine sources
        self.video_sources = Config.VIDEO_SOURCE
        if isinstance(self.video_sources, str):
            self.video_sources = [self.video_sources]
            
        print(f"Initializing trackers for {len(self.video_sources)} sources...")
        self.trackers = {}
        for i in range(len(self.video_sources)):
            self.trackers[i] = BoostTrack(
                reid_weights=Path(Config.REID_MODEL_PATH),
                device=(0 if self.device.type == 'cuda' else 'cpu'),
                half=False,
                with_reid=True,
                max_age=60,
                max_obs=100
            )

        # --- Load ReidExtractor ---
        print("Loading Re-ID Model (OSNet)...")
        # Pass the first tracker to extractor (it shares the model type)
        try:
            self.reid_extractor = ReidExtractor(self.device, self.trackers[0])
        except RuntimeError as e:
            raise RuntimeError(e)

        # --- Load Data ---
        self.load_db()
        
        # State for multi-camera visualization
        self.latest_visualizations = {}

    def advanced_quality_check(self, crop, bbox, frame_shape):
        """Wrapper for QualityChecker.assess_quality()."""
        return self.quality_checker.assess_quality(crop, bbox, frame_shape)

    def load_db(self):
        """Initializes the system by loading the Faiss index."""
        try:
            self.faiss_manager.load_index_from_db()
            next_id = self.mongo_handler.get_next_id()
            self.person_manager.set_next_id(next_id)
            print("System ready. In-memory cache is empty and will be populated on-demand.")
        except Exception as e:
            print(f"!!!! Error during initialization process !!!!: {e}")
            self.faiss_manager.build_index()
            self.known_persons_db.clear()
            self.person_manager.set_next_id(1)

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

    def _verify_and_correct_identities(self):
        """Delegates the identity verification and correction logic."""
        self.identity_verifier.verify_and_correct(
            self.buffer_manager,
            self.faiss_manager,
            self.current_match_threshold
        )

    def _process_tracking_and_reid(self, detections, frame, cam_id):
        """
        Handles BoostTrack update and batch Re-ID feature extraction for a specific camera.
        """
        tracker = self.trackers[cam_id]
        res = tracker.update(detections, frame)
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
                is_good, _, _ = self.quality_checker.assess_quality(crop, box, frame.shape)
                
                if is_good:
                    crops_to_process.append(crop)
                    track_ids_for_crops.append(track_id)
                
                tracker_results_list.append((box, track_id, conf, cls, ind, crop))

        all_vectors = []
        if crops_to_process:
            all_vectors = self.reid_extractor.extract_embeddings_from_crops(crops_to_process)

        vector_map = {}
        if len(all_vectors) == len(track_ids_for_crops):
            vector_map = {tid: vec for tid, vec in zip(track_ids_for_crops, all_vectors)}
        
        return tracker_results_list, vector_map

    def _process_results(self, frame, tracker_results_list, vector_map, cam_id):
        """
        Processes tracker results for matching, enrollment, and updates.
        """
        current_frame_unique_ids = set()
        
        # --- PHASE 1: Reservation ---
        used_pids = set()
        # Pre-scan to lock PIDs owned by existing tracks
        for (_, track_id, _, _, _, _) in tracker_results_list:
            unique_track_id = f"C{cam_id}_T{track_id}"
            existing_pid = self.buffer_manager.track_id_to_permanent_id_map.get(unique_track_id)
            if existing_pid:
                if existing_pid in used_pids:
                    # Conflict: Two tracks claim same PID (rare/legacy)
                    del self.buffer_manager.track_id_to_permanent_id_map[unique_track_id]
                else:
                    used_pids.add(existing_pid)

        # --- PHASE 2: Processing ---
        for (box, track_id, conf, cls, ind, crop) in tracker_results_list:
            unique_track_id = f"C{cam_id}_T{track_id}"
            current_frame_unique_ids.add(unique_track_id)
            current_vector = vector_map.get(track_id)
            
            x1, y1, x2, y2 = map(int, box)
            label, color = f"T{track_id}", (0, 165, 255)
            pid = self.buffer_manager.track_id_to_permanent_id_map.get(unique_track_id)

            # High Quality / Vector Available
            if current_vector is not None:
                if pid:
                    label, color = f"P{pid}", (0, 255, 0)
                    if (time.time() - self.buffer_manager.last_update_time_for_pid.get(pid, 0)) > Config.UPDATE_INTERVAL:
                        self.incremental_update(pid, current_vector)
                        self.buffer_manager.last_update_time_for_pid[pid] = time.time()
                else:
                    now = time.time()
                    # 1. Spatial Match (Cross-camera or within-camera if lost buffer shared)
                    # Note: We share lost_pid_buffer globally.
                    spatial_match_pid, best_match_score = self.spatial_matcher.find_spatial_match(
                        box, self.buffer_manager.lost_pid_buffer, now
                    )
                    
                    if spatial_match_pid is not None and spatial_match_pid in used_pids:
                        spatial_match_pid = None
                    
                    if spatial_match_pid is not None:
                        pid = spatial_match_pid
                        self.buffer_manager.track_id_to_permanent_id_map[unique_track_id] = pid
                        used_pids.add(pid)
                        label = f"P{pid} (ReAcq:{best_match_score:.2f})"
                        color = (0, 255, 128)
                        if pid in self.buffer_manager.lost_pid_buffer:
                            del self.buffer_manager.lost_pid_buffer[pid]
                    else:
                        # 2. Re-ID Match
                        match_id, match_sim = self.find_best_match(current_vector)
                        
                        if match_id is not None and match_id in used_pids:
                            match_id = None
                        
                        if match_id is not None:
                            pid = match_id
                            self.buffer_manager.track_id_to_permanent_id_map[unique_track_id] = pid
                            used_pids.add(pid)
                            label, color = f"P{pid} (S:{match_sim:.2f})", (0, 255, 0)
                        else:
                            # Enrollment
                            is_good, reason, q_score = self.advanced_quality_check(crop, box, frame.shape)
                            buf = self.buffer_manager.active_track_buffers[unique_track_id]
                            
                            buf["vectors"].append(current_vector)
                            buf.setdefault("quality_scores", []).append(q_score)
                            
                            label = f"New? T{track_id} ({len(buf['vectors'])}/{Config.MIN_ENROLL_FRAMES})"
                            buf["last_seen"] = time.time()
                            
                            if len(buf["vectors"]) >= Config.MIN_ENROLL_FRAMES:
                                avg_q = np.mean(buf.get("quality_scores", [0.5]))
                                if avg_q >= 0.65:
                                    new_pid = self.enroll_person(buf["vectors"])
                                    if new_pid:
                                        self.buffer_manager.track_id_to_permanent_id_map[unique_track_id] = new_pid
                                        used_pids.add(new_pid)
                                        if unique_track_id in self.buffer_manager.active_track_buffers:
                                            del self.buffer_manager.active_track_buffers[unique_track_id]
                                else:
                                    buf["vectors"].clear()
                                    buf.get("quality_scores", []).clear()

            # Low Quality / Spatial Only
            else:
                if pid:
                    label, color = f"P{pid} (LowQ)", (100, 255, 100)
                else:
                    now = time.time()
                    spatial_match_pid, best_match_score = self.spatial_matcher.find_spatial_match(
                        box, self.buffer_manager.lost_pid_buffer, now
                    )
                    
                    if spatial_match_pid is not None and spatial_match_pid in used_pids:
                        spatial_match_pid = None
                        
                    if spatial_match_pid is not None:
                        pid = spatial_match_pid
                        self.buffer_manager.track_id_to_permanent_id_map[unique_track_id] = pid
                        used_pids.add(pid)
                        label = f"P{pid} (LowQ-ReAcq)"
                        color = (100, 255, 128)
                        if pid in self.buffer_manager.lost_pid_buffer:
                            del self.buffer_manager.lost_pid_buffer[pid]
                    else:
                        label, color = f"T{track_id} (LowQ)", (0, 100, 155)

            # Update Global DB Last Seen
            if pid is not None:
                self.buffer_manager.last_known_box_for_pid[pid] = box
                if pid not in self.known_persons_db:
                    person_doc = self.mongo_handler.get_person_by_id(pid)
                    if person_doc: self.known_persons_db[pid] = person_doc
                
                if pid in self.known_persons_db:
                    self.known_persons_db[pid]['last_seen_timestamp'] = time.time()

            self.visualizer.draw_person_bbox(frame, (x1, y1, x2, y2), label, color)
        
        return current_frame_unique_ids

    def _perform_cleanup(self, current_frame_unique_ids):
        """
        Performs periodic cleanup. 
        Note: We must only clean up tracks associated with THIS camera's processing 
        OR share the cleanup logic carefully. 
        Currently `current_frame_unique_ids` only contains IDs from the current camera.
        If we clean up everything NOT in this set, we will delete tracks from other cameras!
        
        FIX: We need to know which tracks belong to which camera to cleanup selectively.
        Our unique ID format is `C{cam_id}_T{track_id}`.
        """
        now = time.time()
        self.buffer_manager.cleanup_stale_tracks(now)
        
        # Identify disappeared tracks *for this camera*
        # Filter all known tracks to those belonging to this camera (prefix match)
        # But we don't pass cam_id here. Let's infer or modify.
        # Actually, if we just cleanup based on `last_seen` in `cleanup_stale_tracks`, that handles timeouts.
        # But `disappeared_track_ids` logic below is for IMMEDIATE loss (frame-to-frame).
        # We can't do immediate loss detection easily with async multi-cam unless we track state per cam.
        
        # Simplified: We rely on `cleanup_stale_tracks` (5s timeout) for buffer removal.
        # For "Lost PID" buffer (short-term re-acquisition), we need to know when a track ENDS.
        # BoostTrack handles track death internally? No, we maintain the mapping.
        
        # We will skip the immediate "frame-to-frame" cleanup for now in multi-cam 
        # because we are processing cameras asynchronously.
        # Instead, we rely on the tracker not returning the ID anymore.
        
        # Ideally, we should check: If a track ID (unique) was present recently but not now -> Lost.
        # This requires storing "last_seen_frame" for each unique track.
        pass
        
        self.buffer_manager.cleanup_lost_pids(now)
        if self.buffer_manager.should_run_cleanup(now):
            self.cleanup_stale_buffers(now)

    def run(self):
        """
        The main application loop for multi-camera support.
        """
        self.detector_worker_instance.start()
        print("Waiting for YOLO model to load...")
        self.detector_ready_event.wait()
        print("YOLO ready. Starting streams.")

        caps = []
        for src in self.video_sources:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                print(f"!!!! ERROR: Could not open {src} !!!!")
            caps.append(cap)

        frame_counters = [0] * len(caps)
        self.frame_buffer = {} # (cam_id, frame_id) -> frame

        with torch.inference_mode():
            while True:
                # 1. READ FRAMES
                frames_read = {}
                for i, cap in enumerate(caps):
                    if cap.isOpened():
                        success, frame = cap.read()
                        if success:
                            frames_read[i] = frame
                            frame_counters[i] += 1
                        else:
                            # Loop video for testing purposes? Or stop?
                            # Let's loop for continuous testing
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            success, frame = cap.read()
                            if success:
                                frames_read[i] = frame
                                frame_counters[i] += 1

                if not frames_read:
                    print("All streams ended.")
                    break

                # 2. SEND TO DETECTOR
                for cam_id, frame in frames_read.items():
                    if frame_counters[cam_id] % Config.DETECTION_INTERVAL == 0:
                        frame_id = frame_counters[cam_id]
                        metadata = {'cam_id': cam_id, 'frame_id': frame_id}
                        
                        try:
                            self.frame_queue.put_nowait((metadata, frame.copy()))
                            self.frame_buffer[(cam_id, frame_id)] = frame
                            
                            # Cleanup buffer
                            to_del = [k for k in self.frame_buffer if k[0] == cam_id and k[1] < frame_id - 5]
                            for k in to_del: del self.frame_buffer[k]
                            
                        except queue.Full:
                            pass # Skip frame if detector busy

                # 3. RECEIVE & PROCESS
                try:
                    while not self.detection_queue.empty():
                        metadata, detections = self.detection_queue.get_nowait()
                        res_cam_id = metadata['cam_id']
                        res_frame_id = metadata['frame_id']
                        
                        if (res_cam_id, res_frame_id) in self.frame_buffer:
                            res_frame = self.frame_buffer.pop((res_cam_id, res_frame_id))
                            
                            tracker_res, vector_map = self._process_tracking_and_reid(detections, res_frame, res_cam_id)
                            current_ids = self._process_results(res_frame, tracker_res, vector_map, res_cam_id)
                            
                            self._perform_cleanup(current_ids)
                            
                            # Store for display
                            self.latest_visualizations[res_cam_id] = res_frame
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"Error in processing loop: {e}")

                # 4. VISUALIZE
                display_imgs = []
                for i in range(len(caps)):
                    if i in self.latest_visualizations:
                        display_imgs.append(self.latest_visualizations[i])
                    else:
                        # Placeholder if no frame yet
                        display_imgs.append(np.zeros((480, 640, 3), dtype=np.uint8))
                
                if display_imgs:
                    # Resize to common height
                    h_min = min(img.shape[0] for img in display_imgs)
                    resized_imgs = []
                    for img in display_imgs:
                        if img.shape[0] != h_min:
                            scale = h_min / img.shape[0]
                            w_new = int(img.shape[1] * scale)
                            resized_imgs.append(cv2.resize(img, (w_new, h_min)))
                        else:
                            resized_imgs.append(img)
                    
                    final_display = np.hstack(resized_imgs)
                    
                    # Draw HUD on the combined image (or just top left)
                    # We can use visualizer.draw_hud but it expects 'frame'.
                    # Let's just put simple text.
                    cv2.putText(final_display, f"Cams: {len(caps)} | Persons: {len(self.known_persons_db)}", 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    cv2.imshow("Multi-Camera ReID", final_display)

                if self.handle_keyboard_input() == 'q':
                    break

    def handle_keyboard_input(self):
        """Processes user keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            self.detector_worker_instance.stop()
            return 'q'
        elif key == ord('r'):
            print("RESET...")
            self.person_manager.reset_all()
            self.buffer_manager.reset_all()
            self.faiss_manager.build_index()
            # Clear trackers too?
            # BoostTrack doesn't have easy reset. Maybe recreate them.
            # For now, just reset DB.
        return key
    
    def cleanup(self):
        """Releases all resources gracefully."""
        self.stop_event.set()
        self.detector_worker_instance.stop()
        self.async_writer.shutdown()
        cv2.destroyAllWindows()
        self.mongo_handler.close()
        print("Cleanup complete.")

if __name__ == "__main__":
    app = None
    try:
        app = PersonTrackerApp()
        app.run()
    except Exception as e:
        print(f"!!!! ERROR !!!!: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if app:
            app.cleanup()