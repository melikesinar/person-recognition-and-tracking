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
from utils.logger import get_logger
from core.camera_node import CameraNode
from core.detector import DetectorWorker
from core.reid_extractor import ReidExtractor
from core.quality_checker import QualityChecker
from database.faiss_manager import FaissManager
from database.async_writer import AsyncDBWriter
from database.mongo_handler import MongoHandler
from tracking.person_manager import PersonManager
from tracking.identity_verifier import IdentityVerifier

logger = get_logger()

Config.validate()
Config.print_config()

class MultiCameraSystem:
    """
    Main orchestration class.
    Manages global resources (Database, Detector, ReID) and coordinates CameraNodes.
    """
    def __init__(self):
        logger.info("System initializing (Multi-Camera Mode)...")
        self.stop_event = threading.Event()
        self.paused = False

        # --- Global Shared State ---
        self.match_threshold = Config.MATCH_SIMILARITY_THRESHOLD
        self.detection_threshold_value = multiprocessing.Value('d', Config.DETECTION_SCORE_THRESHOLD)
        self.use_tta = Config.USE_TTA_DEFAULT
        self.debug_spatial = False
        
        # --- Database & Global Components ---
        try:
            self.mongo_handler = MongoHandler()
        except RuntimeError as e:
            logger.critical(f"MongoDB Init Failed: {e}")
            raise RuntimeError(f"MongoDB Init Failed: {e}")

        self.faiss_manager = FaissManager(self.mongo_handler)
        self.async_writer = AsyncDBWriter(
            self.mongo_handler.persons_collection,
            self.mongo_handler.faiss_collection,
            self.stop_event
        )
        self.known_persons_db = {} 
        self.person_manager = PersonManager(
            self.known_persons_db,
            self.async_writer,
            self.faiss_manager
        )
        self.quality_checker = QualityChecker()
        self.global_verifier = IdentityVerifier()
        
        # --- AI Models ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_str = 0 if self.device.type == 'cuda' else 'cpu'
        logger.info(f"Using device: {self.device}")

        logger.info("Loading Re-ID Model...")
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
        
        # Initialize Camera Nodes (Modularly Imported)
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
            logger.info("System ready. DB loaded.")
        except Exception as e:
            logger.error(f"DB Load Error: {e}")
            self.faiss_manager.build_index()
            self.person_manager.set_next_id(1)

    def run(self):
        self.detector_worker.start()
        logger.info("Waiting for Detector...")
        self.detector_ready.wait()
        logger.info("Detector Ready. Starting Streams...")
        
        pause_overlay = None
        
        # Frame Buffer: {cam_id: {frame_id: frame}}
        self.frame_buffers = {node.camera_id: {} for node in self.camera_nodes}
        self.frame_counters = {node.camera_id: 0 for node in self.camera_nodes}

        target_fps = 30.0
        frame_duration = 1.0 / target_fps

        try:
            while True:
                loop_start = time.time()
                
                if not self.paused:
                    # 1. READ FRAMES
                    for node in self.camera_nodes:
                        ret, frame = node.cap.read()
                        if ret:
                            fid = self.frame_counters[node.camera_id]
                            self.frame_counters[node.camera_id] += 1
                            
                            # Store in buffer
                            self.frame_buffers[node.camera_id][fid] = frame.copy()
                            
                            # Send to detector
                            try:
                                self.frame_queue.put_nowait((node.camera_id, fid, frame))
                            except queue.Full:
                                pass 
                        else:
                            node.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    # 2. COLLECT & SYNC DETECTIONS
                    detections_map = {}
                    while not self.detection_queue.empty():
                        try:
                            cam_id, fid, dets = self.detection_queue.get_nowait()
                            if fid in self.frame_buffers[cam_id]:
                                detections_map[cam_id] = (fid, dets)
                        except queue.Empty:
                            break

                    # 3. PROCESS NODES
                    display_frames = []
                    for node in self.camera_nodes:
                        if node.camera_id in detections_map:
                            fid, dets = detections_map[node.camera_id]
                            
                            # Get the EXACT frame
                            frame_to_process = self.frame_buffers[node.camera_id].pop(fid)
                            
                            # Clean up older frames
                            keys_to_remove = [k for k in self.frame_buffers[node.camera_id] if k < fid]
                            for k in keys_to_remove:
                                del self.frame_buffers[node.camera_id][k]
                                
                            annotated_frame = node.process_frame(frame_to_process, dets)
                            node.last_display_frame = annotated_frame
                            
                        if hasattr(node, 'last_display_frame') and node.last_display_frame is not None:
                            display_frames.append(node.last_display_frame)

                    # --- SPATIAL DEBUG VISUALIZATION ---
                    if self.debug_spatial:
                        matcher = self.global_verifier.cross_camera_matcher
                        for node_a in self.camera_nodes:
                            if node_a.camera_id not in detections_map: continue
                            _, dets_a = detections_map[node_a.camera_id]
                            
                            for box_a in dets_a:
                                rect_a = box_a[:4]
                                feet_a = matcher.get_feet_point(rect_a)
                                
                                for node_b in self.camera_nodes:
                                    if node_a.camera_id == node_b.camera_id: continue
                                    
                                    # Check Matrix A -> B
                                    key = f"{node_a.camera_id}-{node_b.camera_id}"
                                    if key in matcher.homography_matrices:
                                        H = np.array(matcher.homography_matrices[key])
                                        proj = matcher.project_point(feet_a, H)
                                        
                                        if proj:
                                            px, py = int(proj[0]), int(proj[1])
                                            # Draw on Node B
                                            if hasattr(node_b, 'last_display_frame') and node_b.last_display_frame is not None:
                                                cv2.circle(node_b.last_display_frame, (px, py), 6, (0, 255, 255), -1)
                                                cv2.circle(node_b.last_display_frame, (px, py), 10, (0, 0, 0), 2)

                    # 4. GLOBAL CHECK
                    self.global_verifier.check_redundant_identities(self.person_manager, self.camera_nodes)

                    # 5. DISPLAY
                    if display_frames:
                        target_h = 480
                        resized_frames = []
                        for f in display_frames:
                            h, w = f.shape[:2]
                            scale = target_h / h
                            new_w = int(w * scale)
                            resized_frames.append(cv2.resize(f, (new_w, target_h)))
                        
                        combined = np.hstack(resized_frames)
                        pause_overlay = combined.copy()
                        cv2.imshow("Multi-Camera Tracking", combined)

                else:
                    if pause_overlay is not None:
                        display = pause_overlay.copy()
                        cv2.putText(display, "PAUSED", (display.shape[1]//2 - 100, display.shape[0]//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.imshow("Multi-Camera Tracking", display)
                    time.sleep(0.1)

                # 6. Global FPS Limiter
                process_time = time.time() - loop_start
                wait_time = frame_duration - process_time
                if wait_time > 0:
                    time.sleep(wait_time)

                key = self.handle_input()
                if key == 'q': break
                    
                self.global_resources['match_threshold'] = self.match_threshold
                self.global_resources['use_tta'] = self.use_tta

        finally:
            self.cleanup()

    def handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): return 'q'
        if key == ord('p'): 
            self.paused = not self.paused
            print(f"System {'PAUSED' if self.paused else 'RESUMED'}")
        if key == ord('u'): self.detection_threshold_value.value = min(1.0, self.detection_threshold_value.value + 0.05)
        if key == ord('j'): self.detection_threshold_value.value = max(0.0, self.detection_threshold_value.value - 0.05)
        if key == ord('y'): self.match_threshold = min(1.0, self.match_threshold + 0.05)
        if key == ord('h'): self.match_threshold = max(0.0, self.match_threshold - 0.05)
        if key == ord('d'):
            self.debug_spatial = not self.debug_spatial
            print(f"Spatial Debug Mode: {'ON' if self.debug_spatial else 'OFF'}")
        if key == ord('t'): 
            self.use_tta = not self.use_tta
            self.reid_extractor.set_tta_status(self.use_tta)
        if key == ord('r'):
            logger.warning("RESET DB TRIGGERED BY USER...")
            self.person_manager.reset_all()
            self.faiss_manager.build_index()
            for node in self.camera_nodes:
                node.buffer_manager.reset_all()
        return key

    def cleanup(self):
        logger.info("System shutting down...")
        self.stop_event.set()
        self.detector_worker.stop()
        self.async_writer.shutdown()
        for node in self.camera_nodes:
            node.cap.release()
        cv2.destroyAllWindows()
        self.mongo_handler.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        app = MultiCameraSystem()
        app.run()
    except Exception as e:
        logger.critical(f"CRITICAL SYSTEM FAILURE: {e}")
        import traceback
        traceback.print_exc()