# main_reid_vectors.py
import cv2
import time
import torch
import queue
import pymongo
import threading
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BoostTrack
from datetime import datetime

# Import modular components
from utils.config import Config
from utils.visualizer import Visualizer
from core.quality_checker import QualityChecker
from core.detector import DetectorWorker
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
        self.current_detection_threshold = Config.DETECTION_SCORE_THRESHOLD

        # TTA settings
        self.USE_TTA = Config.USE_TTA_DEFAULT
        self.TTA_HORIZONTAL_FLIP = Config.TTA_HORIZONTAL_FLIP

        # --- Database Connection ---
        try:
            self.mongo_handler = MongoHandler()
        except RuntimeError:
            exit()  # Exit if connection fails

        # --- Initialize Modular Components ---
        self.faiss_manager = FaissManager(self.mongo_handler.faiss_collection)
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

        # --- Thread Communication (for detector) ---
        self.frame_queue = queue.Queue(maxsize=1)
        self.detection_queue = queue.Queue(maxsize=1)

        # --- Load Models ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- Load Detector ---
        self.detector_worker_instance = DetectorWorker(
            self.frame_queue,
            self.detection_queue,
            self.current_detection_threshold,
            self.device
        )

        print("Loading Re-ID Model (OSNet)...")
        self.tracker = BoostTrack(
            reid_weights=Path(Config.REID_MODEL_PATH),
            device=(0 if self.device.type == 'cuda' else 'cpu'),
            half=False,
            with_reid=True
        )
        if not hasattr(self.tracker, 'reid_model') or self.tracker.reid_model is None:
            raise RuntimeError("Re-ID model failed to load! Exiting.")
        else:
            print("SUCCESS: Re-ID MODEL LOADED.")

        # ImageNet normalization tensors
        self.IMAGENET_MEAN = torch.tensor(Config.IMAGENET_MEAN).view(1, 3, 1, 1)
        self.IMAGENET_STD = torch.tensor(Config.IMAGENET_STD).view(1, 3, 1, 1)

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
        Loads person database from MongoDB and initializes Faiss index.
        """
        try:
            # 1. Load person data from MongoDB into cache
            next_id = self.mongo_handler.load_db_to_cache(self.known_persons_db)
            self.person_manager.set_next_id(next_id)

            # 2. Load or build Faiss index
            self.faiss_manager.load_index_from_db(self.known_persons_db)

        except Exception as e:
            print(f"Error in load_db process: {e}")
            # Ensure a fresh start if loading fails catastrophically
            self.faiss_manager.build_index(self.known_persons_db)
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

    def l2_normalize(self, v):
        """Wrapper for PersonManager.l2_normalize()."""
        return self.person_manager.l2_normalize(v)

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

    def preprocess_crop_advanced(self, crop):
        """
        Applies advanced image preprocessing techniques to a detected person crop
        before feature extraction to enhance image quality and contrast,
        thereby improving the quality of the generated embedding.

        Techniques include Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Args:
            crop (np.ndarray): The BGR image region of the person.

        Returns:
            np.ndarray: The enhanced BGR image crop.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to the L (Lightness) channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def extract_embeddings_from_crops(self, crops):
        """
        Extracts feature embeddings (vectors) for a batch of cropped person images
        using the loaded Re-ID model (OSNet). Includes best practices such as:
        - Advanced preprocessing (CLAHE)
        - ImageNet normalization
        - Test-Time Augmentation (TTA) with horizontal flip for robustness.

        Args:
            crops (list): List of BGR image crops (np.ndarray).

        Returns:
            list: List of L2-normalized feature embeddings (np.ndarray).
        """
        rm = getattr(self.tracker, "reid_model", None)
        if rm is None or not hasattr(rm, 'model') or not callable(rm.model):
            print("!!!! WARNING: Re-ID model is not callable, returning random vectors! !!!!")
            return [self.l2_normalize(np.random.randn(Config.VECTOR_DIMENSION).astype(np.float32)) for _ in crops]

        # Filter and process valid crops
        valid_crops = [c for c in crops if c.size > 0]
        if not valid_crops:
            return []

        # Resize and apply advanced preprocessing
        resized_crops = [
            cv2.resize(
                self.preprocess_crop_advanced(c),
                Config.REID_INPUT_SIZE,
                interpolation=cv2.INTER_LINEAR
            )
            for c in valid_crops
        ]

        # Convert to a batch tensor: [N, H, W, C] -> [N, C, H, W]
        images_np = np.stack(resized_crops, axis=0)
        tensor_batch = torch.from_numpy(images_np).permute(0, 3, 1, 2).float().to(self.device)

        # Normalize to [0, 1] and then apply ImageNet normalization
        tensor_batch = tensor_batch / 255.0

        if self.IMAGENET_MEAN.device != self.device:
            self.IMAGENET_MEAN = self.IMAGENET_MEAN.to(self.device)
            self.IMAGENET_STD = self.IMAGENET_STD.to(self.device)

        tensor_batch = (tensor_batch - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # Extract features with optional TTA (Test-Time Augmentation)
        if self.USE_TTA and self.TTA_HORIZONTAL_FLIP:
            # Average features from the original image and its horizontal flip
            features_original = rm.model(tensor_batch)
            features_flipped = rm.model(torch.flip(tensor_batch, dims=[3]))
            features = (features_original + features_flipped) / 2.0
        else:
            # Standard feature extraction
            features = rm.model(tensor_batch)

        # L2 Normalize and convert to numpy array
        embeddings = [self.l2_normalize(v.cpu().numpy()) for v in features]

        return embeddings



    def run(self):
        """
        The main application loop. It handles video capture, tracking update,
        batch feature extraction, Re-ID matching, spatial re-acquisition,
        enrollment logic, drawing results, and user input processing.
        """
        self.detector_worker_instance.start()

        cap = cv2.VideoCapture(0)  # Assumes default camera
        if not cap.isOpened():
            print("!!!! ERROR: Could not open camera. !!!!")
            return

        # WINDOW_NAME = "ReID Vectors (MongoDB - Faiss)"
        # cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

                # Eşiği 0.0001 saniye olarak belirleyelim
                time_diff = current_time - prev_time
                if time_diff > 0.0001:
                    fps = 1.0 / time_diff
                else:
                    fps = 0.0  # Veya son bilinen FPS değerini koruyun

                prev_time = current_time

                # --- 1. Detection (Asynchronous Input) ---
                if frame_count % Config.DETECTION_INTERVAL == 0:
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        # Skip if detector is still busy
                        pass

                try:
                    # Get the latest detections from the worker thread
                    detections = self.detection_queue.get_nowait()
                except queue.Empty:
                    # Use old detections if new ones aren't ready
                    pass

                # --- 2. Tracking (BoostTrack Update) ---
                res = self.tracker.update(detections, frame)

                # --- 3. Batch Re-Identification Feature Extraction ---
                crops_to_process = []
                track_ids_for_crops = []
                tracker_results_list = []

                # Prepare list of crops for batch processing
                for *box, track_id, conf, cls, ind in res:
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    h, w, _ = frame.shape
                    # Boundary clamping
                    x1 = max(0, min(w - 1, x1))
                    x2 = max(0, min(w - 1, x2))
                    y1 = max(0, min(h - 1, y1))
                    y2 = max(0, min(h - 1, y2))
                    if x2 <= x1 or y2 <= y1: continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops_to_process.append(crop)
                        track_ids_for_crops.append(track_id)
                        tracker_results_list.append((box, track_id, conf, cls, ind, crop))

                all_vectors = []
                if crops_to_process:
                    # Extract all vectors in a single efficient batch
                    all_vectors = self.extract_embeddings_from_crops(crops_to_process)

                vector_map = {}
                if len(all_vectors) == len(track_ids_for_crops):
                    vector_map = {tid: vec for tid, vec in zip(track_ids_for_crops, all_vectors)}
                else:
                    print(
                        f"!!!! WARNING: Mismatch in batch Re-ID. Crops: {len(crops_to_process)}, Vectors: {len(all_vectors)}")

                # --- 4. Process Results (Matching, Enrollment, Update) ---
                current_frame_track_ids = set()
                for (box, track_id, conf, cls, ind, crop) in tracker_results_list:
                    current_frame_track_ids.add(track_id)
                    current_vector = vector_map.get(track_id)
                    if current_vector is None:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    label, color = f"T{track_id}", (0, 165, 255)  # Default: Transient Track ID (Orange)

                    if track_id in self.buffer_manager.track_id_to_permanent_id_map:
                        # Existing known person
                        pid = self.buffer_manager.track_id_to_permanent_id_map[track_id]
                        self.buffer_manager.last_known_box_for_pid[pid] = box
                        label, color = f"P{pid}", (0, 255, 0)  # Green for P-ID match

                        # Incremental master vector update check
                        if (time.time() - self.buffer_manager.last_update_time_for_pid[pid]) > Config.UPDATE_INTERVAL:
                            self.incremental_update(pid, current_vector)
                            self.buffer_manager.last_update_time_for_pid[pid] = time.time()
                    else:
                        # --- ADVANCED Spatial Re-acquisition with Motion Prediction ---
                        now = time.time()
                        spatial_match_pid, best_match_score = self.spatial_matcher.find_spatial_match(
                            box,
                            self.buffer_manager.lost_pid_buffer,
                            now
                        )

                        if spatial_match_pid is not None:
                            # SPATIAL MATCH FOUND: Assign P-ID to the current T-ID
                            self.buffer_manager.track_id_to_permanent_id_map[track_id] = spatial_match_pid
                            label = f"P{spatial_match_pid} (ReAcq:{best_match_score:.2f})"
                            color = (0, 255, 128)  # Greenish for re-acquired

                            # Remove from lost buffer and update position
                            del self.buffer_manager.lost_pid_buffer[spatial_match_pid]
                            self.buffer_manager.last_known_box_for_pid[spatial_match_pid] = box
                            print(f"DEBUG: P{spatial_match_pid} spatially re-acquired by T{track_id}. "
                                  f"Match score: {best_match_score:.2f}")
                        else:
                            # --- Re-ID Match Search (Faiss) ---
                            match_id, match_sim = self.find_best_match(current_vector)

                            if match_id is not None:
                                # Re-ID match found
                                self.buffer_manager.track_id_to_permanent_id_map[track_id] = match_id
                                label, color = f"P{match_id} (S:{match_sim:.2f})", (0, 255, 0)
                            else:
                                # --- Enrollment Candidate Processing ---
                                is_good_quality, reason, quality_score = self.advanced_quality_check(
                                    crop, box, frame.shape
                                )

                                if len(self.known_persons_db) > 0:
                                    print(
                                        f"DEBUG: New person (T{track_id}). Best sim: {match_sim:.4f}. "
                                        f"Quality: {quality_score:.2f} | Reason: {reason}")

                                buf = self.buffer_manager.active_track_buffers[track_id]
                                if is_good_quality:
                                    # Collect vector if quality is sufficient
                                    buf["vectors"].append(current_vector)
                                    buf["quality_scores"] = buf.get("quality_scores", [])
                                    buf["quality_scores"].append(quality_score)

                                    # Display enrollment progress and quality
                                    avg_quality = np.mean(buf["quality_scores"])
                                    label = f"New? T{track_id} ({len(buf['vectors'])}/{Config.MIN_ENROLL_FRAMES}) Q:{avg_quality:.2f}"
                                else:
                                    label = f"New? T{track_id} (REJECT: {reason})"

                                buf["last_seen"] = time.time()

                                # Enrollment decision
                                if len(buf["vectors"]) >= Config.MIN_ENROLL_FRAMES:
                                    avg_quality = np.mean(buf.get("quality_scores", [0.5]))

                                    # Final quality threshold check
                                    if avg_quality >= 0.65:
                                        new_pid = self.enroll_person(buf["vectors"])
                                        if new_pid:
                                            print(f"ENROLLED: New person P{new_pid}. "
                                                  f"Vectors: {len(buf['vectors'])} | Avg Quality: {avg_quality:.2f}")
                                            self.buffer_manager.track_id_to_permanent_id_map[track_id] = new_pid
                                            del self.buffer_manager.active_track_buffers[track_id]
                                    else:
                                        print(f"REJECTED ENROLLMENT: T{track_id} failed quality threshold. "
                                              f"Avg Quality: {avg_quality:.2f} < 0.65. Resetting buffer.")
                                        # Reset buffer to collect better samples
                                        buf["vectors"].clear()
                                        buf["quality_scores"].clear()

                    # Draw bounding box and label
                    self.visualizer.draw_person_bbox(frame, (x1, y1, x2, y2), label, color)

                # --- 5. Cleanup ---
                now = time.time()

                # 1. Clean stale track buffers
                self.buffer_manager.cleanup_stale_tracks(now)

                # 2. Handle disappeared PIDs
                disappeared_track_ids = [
                    tid for tid in self.buffer_manager.track_id_to_permanent_id_map
                    if tid not in current_frame_track_ids
                ]

                for tid in disappeared_track_ids:
                    pid = self.buffer_manager.track_id_to_permanent_id_map.pop(tid, None)
                    if pid and pid in self.buffer_manager.last_known_box_for_pid:
                        self.buffer_manager.add_to_lost_buffer(
                            pid,
                            self.buffer_manager.last_known_box_for_pid[pid],
                            now
                        )

                # 3. Clean expired lost PIDs
                self.buffer_manager.cleanup_lost_pids(now)

                # 4. Periodic buffer cleanup
                if self.buffer_manager.should_run_cleanup(now):
                    self.cleanup_stale_buffers(now)

                # Display HUD
                self.visualizer.draw_hud(
                    frame,
                    fps,
                    self.current_detection_threshold,
                    self.current_match_threshold,
                    len(self.known_persons_db),
                    self.USE_TTA
                )

                cv2.imshow("ReID Vectors (MongoDB - Faiss)", frame)

                key = self.handle_keyboard_input()
                if key == 'q':
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

        # Detection Threshold (U=Up, J=Down)
        elif key == ord('u'):
            self.current_detection_threshold = min(1.0, self.current_detection_threshold + 0.05)
            self.detector_worker_instance.set_detection_threshold(self.current_detection_threshold)
            print(f"Detection Threshold set to: {self.current_detection_threshold:.2f}")
        elif key == ord('j'):
            self.current_detection_threshold = max(0.0, self.current_detection_threshold - 0.05)
            self.detector_worker_instance.set_detection_threshold(self.current_detection_threshold)
            print(f"Detection Threshold set to: {self.current_detection_threshold:.2f}")

        # Match Threshold (Y=Up, H=Down)
        elif key == ord('y'):
            self.current_match_threshold = min(1.0, self.current_match_threshold + 0.05)
            print(f"Match Threshold set to: {self.current_match_threshold:.2f}")
        elif key == ord('h'):
            self.current_match_threshold = max(0.0, self.current_match_threshold - 0.05)
            print(f"Match Threshold set to: {self.current_match_threshold:.2f}")

        # Toggle Test-Time Augmentation (TTA)
        elif key == ord('t'):
            self.USE_TTA = not self.USE_TTA
            status = "ENABLED" if self.USE_TTA else "DISABLED"
            print(f"Test-Time Augmentation (TTA): {status}")

        elif key == ord('r'):
            print("RESET: Queuing MongoDB clear operation...")
            self.person_manager.reset_all()
            self.buffer_manager.reset_all()
            self.faiss_manager.build_index(self.known_persons_db)
            print("Program memory (cache and index) has been cleared.")

        elif key == ord('d'):
            print("\n--- DB STATUS (d) ---")
            print(f"Persons in cache: {len(self.known_persons_db)}")

            stats = self.faiss_manager.get_index_stats()
            print(f"Vectors in Faiss index: {stats['ntotal']}")
            print(f"Index build in progress: {stats['build_in_progress']}")
            print(f"Next ID: {self.person_manager.next_permanent_id}")

            for pid, data in self.known_persons_db.items():
                created_date = data.get('created', 'N/A')
                if isinstance(created_date, datetime):
                    created_date = created_date.strftime("%Y-%m-%d %H:%M:%S")
                print(f"  - P{pid}: {data['count']} vectors, Created: {created_date}")
            print("------------------\n")
            return key

        elif key == ord('k'):
            if self.known_persons_db:
                try:
                    last_id = max(self.known_persons_db.keys())
                    self.person_manager.delete_person(last_id)

                    # Clear associated buffers
                    if last_id in self.buffer_manager.recent_vector_buffers:
                        del self.buffer_manager.recent_vector_buffers[last_id]
                    if last_id in self.buffer_manager.last_update_time_for_pid:
                        del self.buffer_manager.last_update_time_for_pid[last_id]
                    if last_id in self.buffer_manager.last_known_box_for_pid:
                        del self.buffer_manager.last_known_box_for_pid[last_id]

                    self.buffer_manager.active_track_buffers.clear()
                    self.buffer_manager.track_id_to_permanent_id_map.clear()
                    self.buffer_manager.lost_pid_buffer.clear()

                    self.faiss_manager.build_index(self.known_persons_db)
                    print(f"DELETED (k): P{last_id} deleted. Index rebuilt.")
                except Exception as e:
                    print(f"Error deleting person: {e}")
            else:
                print("DB is already empty. Nothing to delete.")
            return key

    def cleanup(self):
        """Releases all resources gracefully."""
        self.stop_event.set()
        print("Cleaning up and exiting...")

        # Shutdown detector worker (if not already stopped by 'q' key)
        self.detector_worker_instance.stop()

        # Shutdown async writer
        self.async_writer.shutdown()

        cv2.destroyAllWindows()
        self.mongo_handler.close()
        print(f"Exited cleanly. DB entries in cache: {len(self.known_persons_db)}")

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
        try:
            if app:
                app.cleanup()
        except NameError:
            cv2.destroyAllWindows()
            print("Application failed to initialize, cleaning up windows.")