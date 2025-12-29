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
# Config.print_config() # Optional: clutter reduction

class CameraNode(threading.Thread):
    """
    Represents a single camera stream processing unit.
    Captures frames, sends them to the shared detector, and processes results.
    """
    def __init__(self, cam_config, shared_components, detector_input_queue, visualizer):
        super().__init__()
        self.cam_id = cam_config["id"]
        self.source = cam_config["source"]
        self.description = cam_config.get("description", self.cam_id)
        
        self.shared = shared_components # shared dict of managers (person, buffer, faiss, etc.)
        self.detector_input_queue = detector_input_queue
        
        # Dedicated queue for receiving detections FOR THIS CAMERA
        self.result_queue = queue.Queue(maxsize=2) 
        
        self.visualizer = visualizer
        self.stop_event = threading.Event()
        
        # Frame Exchange (Thread-Safe)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Local Tracking State
        self.tracker = BoostTrack(
            reid_weights=Path(Config.REID_MODEL_PATH),
            device=self.shared['device_str'],
            half=False,
            with_reid=True,
            max_age=60,
            max_obs=100
        )
        
        # Local ReID Extractor (Lightweight wrapper sharing the model if possible, 
        # but here we instantiate per thread or share model? 
        # OSNet is loaded inside BoostTrack usually, or we can share the reid_extractor object if it's thread-safe.
        # ReidExtractor uses self.tracker.model. Since we have separate trackers, we have separate models.
        # Ideally we should share the model weights to save VRAM, but BoostTrack encapsulates it.
        # For now, we proceed with separate trackers.
        self.reid_extractor = ReidExtractor(self.shared['device_obj'], self.tracker)
        
        # Local Buffer Manager for Track-to-PID mapping specific to this camera view
        # (A person might have Track ID 5 in Cam1 and Track ID 8 in Cam2)
        self.buffer_manager = BufferManager() 
        
        # Local Spatial Matcher
        self.spatial_matcher = SpatialMatcher()

    def run(self):
        print(f"[{self.cam_id}] Connecting to source: {self.source}")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            # Fallback for integer sources (webcams) passed as strings
            if isinstance(self.source, str) and self.source.isdigit():
                 cap = cv2.VideoCapture(int(self.source))
            
            if not cap.isOpened():
                print(f"[{self.cam_id}] !!!! ERROR: Could not open source !!!!")
                return

        frame_count = 0
        prev_time = 0
        
        detections = np.empty((0, 6), dtype=np.float32)

        while not self.stop_event.is_set():
            success, frame = cap.read()
            if not success:
                print(f"[{self.cam_id}] Stream ended.")
                # Loop video for testing purposes?
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # 1. Send to Shared Detector
            if frame_count % Config.DETECTION_INTERVAL == 0:
                try:
                    # Send (cam_id, frame)
                    self.detector_input_queue.put_nowait((self.cam_id, frame.copy()))
                except queue.Full:
                    pass # Skip detection if detector is busy
            
            # 2. Check for results
            try:
                # Get latest available detection
                while not self.result_queue.empty():
                    detections = self.result_queue.get_nowait()
            except queue.Empty:
                pass

            # 3. Process Tracking & ReID
            # Note: We use the *shared* PersonManager and FaissManager here!
            tracker_results_list, vector_map = self._process_tracking_and_reid(detections, frame)
            
            current_frame_track_ids = self._process_results(frame, tracker_results_list, vector_map)

            # 4. Cleanup & Vis
            self._perform_cleanup(current_frame_track_ids)
            
            self.visualizer.draw_hud(
                frame, fps, self.shared['detection_threshold'].value,
                self.shared['match_threshold'], 
                len(self.shared['known_persons_db']), # Global count
                self.shared['use_tta']
            )
            
            # Update Shared Frame safely
            with self.frame_lock:
                self.latest_frame = frame.copy()
            
            # We need a way to break. Main thread handles 'q'.
            if self.stop_event.is_set():
                break
                
            # Rate limit for stability
            # time.sleep(0.001) 

        cap.release()
        print(f"[{self.cam_id}] Stopped.")

    def _process_tracking_and_reid(self, detections, frame):
        # ... logic similar to original, but using local tracker ...
        res = self.tracker.update(detections, frame)
        crops, tids, results = [], [], []
        
        for *box, track_id, conf, cls, ind in res:
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            h, w, _ = frame.shape
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1: continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                is_good, _, _ = self.shared['quality_checker'].assess_quality(crop, box, frame.shape)
                if is_good:
                    crops.append(crop)
                    tids.append(track_id)
                results.append((box, track_id, conf, cls, ind, crop))

        all_vectors = []
        if crops:
            all_vectors = self.reid_extractor.extract_embeddings_from_crops(crops)
        
        vec_map = {tid: vec for tid, vec in zip(tids, all_vectors)} if len(all_vectors) == len(tids) else {}
        return results, vec_map

    def _process_results(self, frame, tracker_results, vector_map):
        # ... logic similar to original, but using shared PersonManager ...
        current_tids = set()
        used_pids = set() # Per-frame uniqueness constraint

        # Phase 1: Reservation
        for (_, tid, _, _, _, _) in tracker_results:
            existing_pid = self.buffer_manager.track_id_to_permanent_id_map.get(tid)
            if existing_pid:
                if existing_pid in used_pids:
                    del self.buffer_manager.track_id_to_permanent_id_map[tid]
                else:
                    used_pids.add(existing_pid)

        # Phase 2: Assignment
        for (box, tid, conf, cls, ind, crop) in tracker_results:
            current_tids.add(tid)
            vec = vector_map.get(tid)
            pid = self.buffer_manager.track_id_to_permanent_id_map.get(tid)
            label, color = f"T{tid}", (0, 165, 255)

            if vec is not None:
                if pid:
                    # Existing Global ID
                    label, color = f"G-ID:{pid}", (0, 255, 0)
                    if (time.time() - self.buffer_manager.last_update_time_for_pid.get(pid, 0)) > Config.UPDATE_INTERVAL:
                        # Shared PersonManager Update
                        self.shared['person_manager'].incremental_update(pid, vec)
                        self.buffer_manager.last_update_time_for_pid[pid] = time.time()
                else:
                    # Identify
                    # 1. Spatial (Local buffer only)
                    spatial_pid, score = self.spatial_matcher.find_spatial_match(
                        box, self.buffer_manager.lost_pid_buffer, time.time()
                    )
                    if spatial_pid and spatial_pid in used_pids: spatial_pid = None

                    if spatial_pid:
                        pid = spatial_pid
                        self.buffer_manager.track_id_to_permanent_id_map[tid] = pid
                        used_pids.add(pid)
                        label, color = f"G-ID:{pid} (Sp)", (0, 255, 128)
                        if pid in self.buffer_manager.lost_pid_buffer: del self.buffer_manager.lost_pid_buffer[pid]
                    else:
                        # 2. Re-ID (Global Search)
                        match_id, match_sim = self.shared['faiss_manager'].search(vec, self.shared['match_threshold'])
                        if match_id and match_id in used_pids: match_id = None
                        
                        if match_id:
                            pid = match_id
                            self.buffer_manager.track_id_to_permanent_id_map[tid] = pid
                            used_pids.add(pid)
                            label, color = f"G-ID:{pid} ({match_sim:.2f})", (0, 255, 0)
                        else:
                            # Enroll
                            buf = self.buffer_manager.active_track_buffers[tid]
                            buf["vectors"].append(vec)
                            if len(buf["vectors"]) >= Config.MIN_ENROLL_FRAMES:
                                # Global Enrollment
                                new_pid = self.shared['person_manager'].enroll_person(buf["vectors"])
                                if new_pid:
                                    self.buffer_manager.track_id_to_permanent_id_map[tid] = new_pid
                                    used_pids.add(new_pid)
                                    if tid in self.buffer_manager.active_track_buffers:
                                        del self.buffer_manager.active_track_buffers[tid]
            else:
                # Low Quality / No Vec
                if pid:
                    label, color = f"G-ID:{pid} (LowQ)", (100, 255, 100)
            
            # Common updates
            if pid:
                self.buffer_manager.last_known_box_for_pid[pid] = box
                # Update global cache locally if needed, but PersonManager handles DB
                # We can update shared db cache for UI stats
                if pid not in self.shared['known_persons_db']:
                     doc = self.shared['mongo_handler'].get_person_by_id(pid)
                     if doc: self.shared['known_persons_db'][pid] = doc
            
            self.visualizer.draw_person_bbox(frame, (int(box[0]), int(box[1]), int(box[2]), int(box[3])), label, color)
        
        return current_tids

    def _perform_cleanup(self, current_tids):
        now = time.time()
        self.buffer_manager.cleanup_stale_tracks(now)
        # Handle lost tracks -> move to lost buffer for spatial matching
        disappeared = [tid for tid in self.buffer_manager.track_id_to_permanent_id_map if tid not in current_tids]
        for tid in disappeared:
            pid = self.buffer_manager.track_id_to_permanent_id_map.pop(tid, None)
            if pid and pid in self.buffer_manager.last_known_box_for_pid:
                self.buffer_manager.add_to_lost_buffer(pid, self.buffer_manager.last_known_box_for_pid[pid], now)
        
        self.buffer_manager.cleanup_lost_pids(now)


class PersonTrackerApp:
    def __init__(self):
        print("System initializing Multi-Camera Node Architecture...")
        self.stop_event = threading.Event()
        
        # --- Shared State ---
        self.known_persons_db = {}
        self.detection_threshold_value = multiprocessing.Value('d', Config.DETECTION_SCORE_THRESHOLD)
        self.match_threshold = Config.MATCH_SIMILARITY_THRESHOLD
        self.use_tta = Config.USE_TTA_DEFAULT
        
        # --- Shared Database & Logic ---
        self.mongo_handler = MongoHandler()
        self.faiss_manager = FaissManager(self.mongo_handler)
        self.async_writer = AsyncDBWriter(
            self.mongo_handler.persons_collection,
            self.mongo_handler.faiss_collection,
            self.stop_event
        )
        self.person_manager = PersonManager(
            self.known_persons_db, self.async_writer, self.faiss_manager
        )
        self.quality_checker = QualityChecker()
        self.visualizer = Visualizer()

        # --- Device Setup ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_str = 0 if self.device.type == 'cuda' else 'cpu'

        # --- Detector Logic ---
        # Queue: (cam_id, frame)
        self.detector_input_queue = multiprocessing.Queue(maxsize=len(Config.VIDEO_SOURCES) * 2)
        # Queue: (cam_id, detections)
        self.detector_output_queue = multiprocessing.Queue(maxsize=len(Config.VIDEO_SOURCES) * 2)
        self.detector_ready_event = multiprocessing.Event()
        
        self.detector_worker = DetectorWorker(
            self.detector_input_queue,
            self.detector_output_queue,
            self.detection_threshold_value,
            self.device_str,
            self.detector_ready_event
        )

        # --- Load Data ---
        self.load_db()

        # --- Initialize Camera Nodes ---
        self.camera_nodes = []
        self.shared_components = {
            'person_manager': self.person_manager,
            'faiss_manager': self.faiss_manager,
            'mongo_handler': self.mongo_handler,
            'quality_checker': self.quality_checker,
            'known_persons_db': self.known_persons_db,
            'detection_threshold': self.detection_threshold_value,
            'match_threshold': self.match_threshold,
            'use_tta': self.use_tta,
            'device_obj': self.device,
            'device_str': self.device_str
        }

    def load_db(self):
        try:
            self.faiss_manager.load_index_from_db()
            self.person_manager.set_next_id(self.mongo_handler.get_next_id())
            print("Database loaded. Ready for multi-camera fusion.")
        except Exception as e:
            print(f"Init Error: {e}")
            self.faiss_manager.build_index()
            self.person_manager.set_next_id(1)

    def run(self):
        # 1. Start Detector
        self.detector_worker.start()
        print("Waiting for Detector...")
        self.detector_ready_event.wait()
        
        # 2. Start Camera Nodes
        for cam_config in Config.VIDEO_SOURCES:
            node = CameraNode(
                cam_config, 
                self.shared_components, 
                self.detector_input_queue,
                self.visualizer
            )
            node.start()
            self.camera_nodes.append(node)
        
        print(f"Started {len(self.camera_nodes)} camera nodes.")
        print("Press 'q' to quit.")

        # 3. Main Dispatcher Loop
        # Reads from single detector output and routes to correct camera node
        try:
            while True:
                # Dispatcher Logic
                try:
                    # Get result from detector (cam_id, result)
                    cam_id, detections = self.detector_output_queue.get(timeout=0.01)
                    
                    # Find target node
                    target_node = next((n for n in self.camera_nodes if n.cam_id == cam_id), None)
                    if target_node:
                        try:
                            target_node.result_queue.put_nowait(detections)
                        except queue.Full:
                            pass # Drop if camera thread is lagging
                except queue.Empty:
                    pass

                # --- Visualization (Main Thread) ---
                for node in self.camera_nodes:
                    frame_to_show = None
                    with node.frame_lock:
                        if node.latest_frame is not None:
                            frame_to_show = node.latest_frame
                    
                    if frame_to_show is not None:
                        # Resize for better tiling if needed, for now just show
                        cv2.imshow(f"Cam: {node.cam_id}", frame_to_show)

                # Handle Global Keyboard Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopping...")
                    break
                elif key == ord('r'):
                     print("Resetting DB...")
                     self.person_manager.reset_all()
                     self.faiss_manager.build_index()
        
        except KeyboardInterrupt:
            print("Interrupted.")
        
        finally:
            self.cleanup()

    def cleanup(self):
        print("Stopping nodes...")
        for node in self.camera_nodes:
            node.stop_event.set()
        
        for node in self.camera_nodes:
            node.join()
            
        self.stop_event.set()
        self.detector_worker.stop()
        self.async_writer.shutdown()
        self.mongo_handler.close()
        cv2.destroyAllWindows()
        print("System shutdown complete.")

if __name__ == "__main__":
    app = PersonTrackerApp()
    app.run()
