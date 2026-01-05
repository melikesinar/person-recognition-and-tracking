# main_multi_camera.py
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

class CameraNode:
    """
    Represents a completely independent tracking system for a SINGLE camera source.
    """
    def __init__(self, camera_id, video_source, global_resources):
        self.camera_id = camera_id
        self.video_source = video_source
        self.resources = global_resources 
        
        print(f"Initializing CameraNode {camera_id} for source: {video_source}")
        
        # --- Independent State Components ---
        self.buffer_manager = BufferManager()
        self.spatial_matcher = SpatialMatcher()
        self.visualizer = Visualizer()
        self.identity_verifier = IdentityVerifier()
        
        # --- Independent Tracker Instance ---
        self.tracker = BoostTrack(
            reid_weights=Path(Config.REID_MODEL_PATH),
            device=(0 if self.resources['device'].type == 'cuda' else 'cpu'),
            half=False,
            with_reid=True,
            max_age=60,
            max_obs=100
        )
        
        # Video Capture
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"!!!! ERROR: Could not open video source for Cam {camera_id}: {video_source}")
        
        self.frame_count = 0
        self.prev_time = time.time()
        self.fps = 0.0
        
        # FPS Limiter Settings
        self.target_fps = 30.0
        self.frame_duration = 1.0 / self.target_fps

    def process_frame(self, frame, detections):
        start_time = time.time()
        self.frame_count += 1
        
        # 1. Update Tracker & Extract Crops
        tracker_results_list, vector_map = self._process_tracking_and_reid(detections, frame)
        
        # 2. Process Identifies (Match/Enroll)
        current_frame_track_ids = self._process_results(frame, tracker_results_list, vector_map)
        
        # 3. Verify & Cleanup
        if self.frame_count % Config.ID_VERIFICATION_INTERVAL == 0:
            self._verify_and_correct_identities()
            
        self._perform_cleanup(current_frame_track_ids)
        
        # 4. FPS Limiter
        process_time = time.time() - start_time
        wait_time = self.frame_duration - process_time
        if wait_time > 0:
            time.sleep(wait_time)
        
        # 5. Visualization (FPS Calc)
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        if time_diff > 0.0001:
            self.fps = 1.0 / time_diff
        self.prev_time = curr_time
        
        self.visualizer.draw_hud(
            frame, self.fps, self.resources['detection_threshold'].value,
            self.resources['match_threshold'], 
            len(self.resources['person_manager'].known_persons_db), 
            self.resources['use_tta']
        )
        
        # Add Camera ID to HUD
        cv2.putText(frame, f"CAM {self.camera_id}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame

    def _process_tracking_and_reid(self, detections, frame):
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
                is_good, _, _ = self.resources['quality_checker'].assess_quality(crop, box, frame.shape)
                if is_good:
                    crops_to_process.append(crop)
                    track_ids_for_crops.append(track_id)
                tracker_results_list.append((box, track_id, conf, cls, ind, crop))

        all_vectors = []
        if crops_to_process:
            all_vectors = self.resources['reid_extractor'].extract_embeddings_from_crops(crops_to_process)

        vector_map = {}
        if len(all_vectors) == len(track_ids_for_crops):
            vector_map = {tid: vec for tid, vec in zip(track_ids_for_crops, all_vectors)}
        
        return tracker_results_list, vector_map

    def _process_results(self, frame, tracker_results_list, vector_map):
        current_frame_track_ids = set()
        used_pids = set()
        
        # Phase 1: Reservation
        for (_, track_id, _, _, _, _) in tracker_results_list:
            existing_pid = self.buffer_manager.track_id_to_permanent_id_map.get(track_id)
            if existing_pid:
                if existing_pid in used_pids:
                    del self.buffer_manager.track_id_to_permanent_id_map[track_id]
                else:
                    used_pids.add(existing_pid)

        # Phase 2: Assignment
        for (box, track_id, conf, cls, ind, crop) in tracker_results_list:
            current_frame_track_ids.add(track_id)
            current_vector = vector_map.get(track_id)
            
            x1, y1, x2, y2 = map(int, box)
            label = f"T{track_id}"
            color = (0, 165, 255)
            pid = self.buffer_manager.track_id_to_permanent_id_map.get(track_id)

            if current_vector is not None:
                if pid:
                    label, color = f"P{pid}", (0, 255, 0)
                    last_update = self.buffer_manager.last_update_time_for_pid.get(pid, 0)
                    if (time.time() - last_update) > Config.UPDATE_INTERVAL:
                        self.resources['person_manager'].incremental_update(pid, current_vector)
                        self.buffer_manager.last_update_time_for_pid[pid] = time.time()
                else:
                    # New Track -> Try Spatial -> Try ReID -> Enroll
                    now = time.time()
                    spatial_pid, score = self.spatial_matcher.find_spatial_match(
                        box, self.buffer_manager.lost_pid_buffer, now
                    )
                    
                    if spatial_pid and spatial_pid in used_pids: spatial_pid = None
                    
                    if spatial_pid:
                        pid = spatial_pid
                        self.buffer_manager.track_id_to_permanent_id_map[track_id] = pid
                        used_pids.add(pid)
                        label = f"P{pid} (RA)"
                        color = (0, 255, 128)
                        if pid in self.buffer_manager.lost_pid_buffer:
                            del self.buffer_manager.lost_pid_buffer[pid]
                    else:
                        # ReID Match (GLOBAL SEARCH)
                        match_id, match_sim = self.resources['faiss_manager'].search(
                            current_vector, self.resources['match_threshold']
                        )
                        
                        if match_id and match_id in used_pids: match_id = None
                        
                        if match_id:
                            pid = match_id
                            self.buffer_manager.track_id_to_permanent_id_map[track_id] = pid
                            used_pids.add(pid)
                            label, color = f"P{pid} (S:{match_sim:.2f})", (0, 255, 0)
                        else:
                            # Enrollment (GLOBAL ENROLLMENT)
                            is_good, _, q_score = self.resources['quality_checker'].assess_quality(crop, box, frame.shape)
                            buf = self.buffer_manager.active_track_buffers[track_id]
                            
                            buf["vectors"].append(current_vector)
                            buf.setdefault("quality_scores", []).append(q_score)
                            
                            label = f"New? T{track_id} ({len(buf['vectors'])})"
                            buf["last_seen"] = time.time()
                            
                            if len(buf["vectors"]) >= Config.MIN_ENROLL_FRAMES:
                                avg_q = np.mean(buf.get("quality_scores", [0.5]))
                                if avg_q >= 0.65:
                                    new_pid = self.resources['person_manager'].enroll_person(buf["vectors"])
                                    if new_pid:
                                        self.buffer_manager.track_id_to_permanent_id_map[track_id] = new_pid
                                        used_pids.add(new_pid)
                                        if track_id in self.buffer_manager.active_track_buffers:
                                            del self.buffer_manager.active_track_buffers[track_id]
                                else:
                                    buf["vectors"].clear(); buf["quality_scores"].clear()
            else:
                # Low Quality logic
                if pid:
                    label, color = f"P{pid} (LQ)", (100, 255, 100)
                else:
                    now = time.time()
                    spatial_pid, _ = self.spatial_matcher.find_spatial_match(
                        box, self.buffer_manager.lost_pid_buffer, now
                    )
                    if spatial_pid and spatial_pid in used_pids: spatial_pid = None
                    if spatial_pid:
                        pid = spatial_pid
                        self.buffer_manager.track_id_to_permanent_id_map[track_id] = pid
                        used_pids.add(pid)
                        label = f"P{pid} (LQ)", (100, 255, 128)
                        color = (100, 255, 128)
                        if pid in self.buffer_manager.lost_pid_buffer:
                            del self.buffer_manager.lost_pid_buffer[pid]
                    else:
                        label, color = f"T{track_id} (LQ)", (0, 100, 155)

            # Common Updates
            if pid:
                self.buffer_manager.last_known_box_for_pid[pid] = box
                if pid not in self.resources['person_manager'].known_persons_db:
                    p_doc = self.resources['mongo_handler'].get_person_by_id(pid)
                    if p_doc: self.resources['person_manager'].known_persons_db[pid] = p_doc
                if pid in self.resources['person_manager'].known_persons_db:
                    self.resources['person_manager'].known_persons_db[pid]['last_seen_timestamp'] = time.time()

            self.visualizer.draw_person_bbox(frame, (x1, y1, x2, y2), label, color)
        return current_frame_track_ids

    def _verify_and_correct_identities(self):
        self.identity_verifier.verify_and_correct(
            self.buffer_manager,
            self.resources['faiss_manager'],
            self.resources['match_threshold']
        )

    def _perform_cleanup(self, current_frame_track_ids):
        now = time.time()
        self.buffer_manager.cleanup_stale_tracks(now)
        
        disappeared = [tid for tid in self.buffer_manager.track_id_to_permanent_id_map if tid not in current_frame_track_ids]
        for tid in disappeared:
            pid = self.buffer_manager.track_id_to_permanent_id_map.pop(tid, None)
            self.identity_verifier.clear_state_for_track(tid)
            if pid and pid in self.buffer_manager.last_known_box_for_pid:
                self.buffer_manager.add_to_lost_buffer(pid, self.buffer_manager.last_known_box_for_pid[pid], now)

        self.buffer_manager.cleanup_lost_pids(now)
        if self.buffer_manager.should_run_cleanup(now):
            self.buffer_manager.cleanup_stale_buffers(now, self.resources['person_manager'].known_persons_db)


class MultiCameraSystem:
    def __init__(self):
        print("System initializing (Multi-Camera Mode)...")
        self.stop_event = threading.Event()

        # --- Global Shared State ---
        self.match_threshold = Config.MATCH_SIMILARITY_THRESHOLD
        self.detection_threshold_value = multiprocessing.Value('d', Config.DETECTION_SCORE_THRESHOLD)
        self.use_tta = Config.USE_TTA_DEFAULT
        
        # --- Database & Global Components ---
        try:
            self.mongo_handler = MongoHandler()
        except RuntimeError as e:
            raise RuntimeError(f"MongoDB Init Failed: {e}")

        self.faiss_manager = FaissManager(self.mongo_handler)
        self.async_writer = AsyncDBWriter(
            self.mongo_handler.persons_collection,
            self.mongo_handler.faiss_collection,
            self.stop_event
        )
        # Global cache of "Known People" - shared across cameras
        self.known_persons_db = {} 
        self.person_manager = PersonManager(
            self.known_persons_db,
            self.async_writer,
            self.faiss_manager
        )
        self.quality_checker = QualityChecker()
        
        # --- AI Models ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = 0 if self.device.type == 'cuda' else 'cpu'
        print(f"Using device: {self.device}")

        # Shared ReID
        print("Loading Re-ID Model...")
        self.master_tracker = BoostTrack(
            reid_weights=Path(Config.REID_MODEL_PATH),
            device=device_str,
            half=False,
            with_reid=True
        )
        self.reid_extractor = ReidExtractor(self.device, self.master_tracker)

        # Shared Detector
        self.frame_queue = multiprocessing.Queue(maxsize=len(Config.VIDEO_SOURCES) * 2)
        self.detection_queue = multiprocessing.Queue(maxsize=len(Config.VIDEO_SOURCES) * 2)
        self.detector_ready = multiprocessing.Event()
        
        self.detector_worker = DetectorWorker(
            self.frame_queue, self.detection_queue, 
            self.detection_threshold_value, device_str, self.detector_ready
        )

        # --- Resources Dict ---
        self.global_resources = {
            'person_manager': self.person_manager,
            'faiss_manager': self.faiss_manager,
            'mongo_handler': self.mongo_handler,
            'reid_extractor': self.reid_extractor,
            'quality_checker': self.quality_checker,
            'detection_threshold': self.detection_threshold_value,
            'match_threshold': self.match_threshold,
            'use_tta': self.use_tta,
            'device': self.device
        }
        
        # --- Initialize Camera Nodes ---
        self.camera_nodes = []
        for i, src in enumerate(Config.VIDEO_SOURCES):
            node = CameraNode(i, src, self.global_resources)
            self.camera_nodes.append(node)

        self.load_db()

    def load_db(self):
        try:
            self.faiss_manager.load_index_from_db()
            next_id = self.mongo_handler.get_next_id()
            self.person_manager.set_next_id(next_id)
            print("System ready. DB loaded.")
        except Exception as e:
            print(f"DB Load Error: {e}")
            self.faiss_manager.build_index()
            self.person_manager.set_next_id(1)

    def run(self):
        self.detector_worker.start()
        print("Waiting for Detector...")
        self.detector_ready.wait()
        print("Detector Ready. Starting Streams...")

        try:
            while True:
                # 1. Read Frames from all cameras
                frames = {}
                for node in self.camera_nodes:
                    ret, frame = node.cap.read()
                    if ret:
                        frames[node.camera_id] = frame
                        try:
                            self.frame_queue.put_nowait((node.camera_id, frame.copy()))
                        except queue.Full:
                            pass 
                    else:
                        # Auto-rewind for testing
                        node.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                # 2. Collect Detections
                detections_map = {}
                while not self.detection_queue.empty():
                    try:
                        cam_id, dets = self.detection_queue.get_nowait()
                        detections_map[cam_id] = dets
                    except queue.Empty:
                        break

                # 3. Process Per Camera
                display_frames = []
                for node in self.camera_nodes:
                    if node.camera_id in frames:
                        frame = frames[node.camera_id]
                        dets = detections_map.get(node.camera_id, np.empty((0, 6)))
                        
                        annotated_frame = node.process_frame(frame, dets)
                        display_frames.append(annotated_frame)

                # 4. Display Combined (Resize to fixed height for consistent stack)
                if display_frames:
                    target_h = 480
                    resized_frames = []
                    for f in display_frames:
                        h, w = f.shape[:2]
                        scale = target_h / h
                        new_w = int(w * scale)
                        resized_frames.append(cv2.resize(f, (new_w, target_h)))
                    
                    combined = np.hstack(resized_frames)
                    cv2.imshow("Multi-Camera Tracking", combined)

                key = self.handle_input()
                if key == 'q':
                    break
                    
                self.global_resources['match_threshold'] = self.match_threshold
                self.global_resources['use_tta'] = self.use_tta

        finally:
            self.cleanup()

    def handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): return 'q'
        if key == ord('u'): self.detection_threshold_value.value = min(1.0, self.detection_threshold_value.value + 0.05)
        if key == ord('j'): self.detection_threshold_value.value = max(0.0, self.detection_threshold_value.value - 0.05)
        if key == ord('y'): self.match_threshold = min(1.0, self.match_threshold + 0.05)
        if key == ord('h'): self.match_threshold = max(0.0, self.match_threshold - 0.05)
        if key == ord('t'): 
            self.use_tta = not self.use_tta
            self.reid_extractor.set_tta_status(self.use_tta)
        if key == ord('r'):
            print("RESET DB...")
            self.person_manager.reset_all()
            self.faiss_manager.build_index()
            for node in self.camera_nodes:
                node.buffer_manager.reset_all()
        return key

    def cleanup(self):
        self.stop_event.set()
        self.detector_worker.stop()
        self.async_writer.shutdown()
        for node in self.camera_nodes:
            node.cap.release()
        cv2.destroyAllWindows()
        self.mongo_handler.close()

if __name__ == "__main__":
    try:
        app = MultiCameraSystem()
        app.run()
    except Exception as e:
        print(f"CRITICAL: {e}")
        import traceback
        traceback.print_exc()
