# main_reid_vectors.py
import os
import cv2
import time
import faiss
import queue
import torch
import pymongo
import threading
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BoostTrack
from datetime import datetime
from collections import deque
from bson.binary import Binary
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
# Database Configuration
MONGO_CONNECTION_STRING = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("COLLECTION")
MONGO_FAISS_COLLECTION_NAME = os.getenv("FAISS_COLLECTION")
FAISS_INDEX_KEY = "primary_person_index"
TEMP_FAISS_FILENAME = "temp_faiss.index"

# Default and Environment Config Values
DEFAULT_DETECTION_SCORE = 0.7
DEFAULT_MATCH_THRESHOLD = 0.85

DETECTION_SCORE_THRESHOLD = float(os.getenv("DETECTION_SCORE_THRESHOLD", DEFAULT_DETECTION_SCORE))
MATCH_SIMILARITY_THRESHOLD = float(os.getenv("MATCH_SIMILARITY_THRESHOLD", DEFAULT_MATCH_THRESHOLD))
MIN_ENROLL_FRAMES = int(os.getenv("MIN_ENROLL_FRAMES", "30"))  # Minimum frames required to enroll a new person
MIN_BBOX_AREA_ENROLL = int(os.getenv("MIN_BBOX_AREA_ENROLL", "10000"))  # Minimum bounding box area for enrollment
MIN_SHARPNESS_ENROLL = float(
    os.getenv("MIN_SHARPNESS_ENROLL", "50.0"))  # Minimum sharpness (Laplacian var) for enrollment
MAX_VECTORS_FOR_UPDATE = int(
    os.getenv("MAX_VECTORS_FOR_UPDATE", "300"))  # Vector count limit before locking master vector
RECENT_VECTOR_BUFFER_SIZE = int(os.getenv("RECENT_VECTOR_BUFFER_SIZE", "50"))
DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL", "2"))  # Run detection every N frames
REID_INPUT_SIZE = (128, 256)
VECTOR_DIMENSION = 512
SHORT_TERM_MEMORY_TTL = 2.0  # Time in seconds a lost PID stays in buffer
SPATIAL_MATCH_THRESHOLD = 150  # Deprecated, now adaptive


class PersonTrackerApp:
    """
    Main class that encapsulates all logic for the person Re-ID and tracking system,
    including thread management, database operations, and live processing.
    """

    def __init__(self):
        """
        Initializes the class, connects to the database, loads machine learning models (YOLO, OSNet),
        and prepares the initial state, including starting the asynchronous DB writer.
        """
        print("System initializing...")

        # --- State Variables ---
        self.faiss_index = None  # Faiss HNSW index object
        self.faiss_lock = threading.Lock()  # Lock for thread-safe access to Faiss index and map
        self.known_persons_db = {}  # Local cache of enrolled persons (master vectors, counts, etc.)
        self.index_to_pid_map = []  # Map from Faiss index position to permanent_id (P-ID)
        self.lost_pid_buffer = {}  # Buffer for recently lost P-IDs (for re-acquisition)
        self.last_known_box_for_pid = {}  # Stores the last known bounding box for each P-ID
        self.next_permanent_id = 1  # Next available P-ID for enrollment
        self.index_build_in_progress = False  # Flag to prevent multiple index rebuilds simultaneously
        self.pending_enrollments = []  # List of new P-IDs waiting for index rebuild
        self.last_index_rebuild = time.time()
        self.MIN_REBUILD_INTERVAL = 30  # Minimum seconds between forced index rebuilds
        self.BATCH_SIZE_TRIGGER = 5  # Number of new enrollments to trigger a rebuild

        self.stop_event = threading.Event()  # Signal for thread termination

        # Asynchronous DB Writer setup
        self.db_write_queue = queue.Queue()
        self.db_writer_thread = threading.Thread(target=self._db_writer_worker, daemon=True)
        self.db_writer_thread.start()
        print("Async database writer started.")

        # Live tuning thresholds (modified by keyboard presses)
        self.current_match_threshold = MATCH_SIMILARITY_THRESHOLD
        self.current_detection_threshold = DETECTION_SCORE_THRESHOLD

        # --- Thread Communication Queues ---
        self.frame_queue = queue.Queue(maxsize=1)  # Frame buffer for detector thread
        self.detection_queue = queue.Queue(maxsize=1)  # Detection results from detector thread

        # --- Tracking and Enrollment Buffers ---
        self.active_track_buffers = defaultdict(lambda: {
            "vectors": [],  # Temporary vector storage for new tracks
            "crops": [],
            "quality_scores": [],
            "last_seen": time.time()
        })
        self.track_id_to_permanent_id_map = {}  # Map from transient BoostTrack ID (T-ID) to P-ID
        self.recent_vector_buffers = defaultdict(
            lambda: deque(maxlen=RECENT_VECTOR_BUFFER_SIZE))  # Buffer for incremental updates
        self.last_update_time_for_pid = defaultdict(float)
        self.UPDATE_INTERVAL = 1.0  # Minimum seconds between master vector updates for a P-ID
        self.BUFFER_CLEANUP_INTERVAL = 60.0  # Time between main buffer cleanup checks
        self.BUFFER_MAX_IDLE_TIME = 3600.0  # Time after which a buffer entry can be considered stale
        self.last_buffer_cleanup = time.time()

        # ImageNet normalization constants for Re-ID model
        self.IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # TTA (Test-Time Augmentation) settings for Re-ID feature extraction
        self.USE_TTA = True
        self.TTA_HORIZONTAL_FLIP = True
        self.TTA_MULTI_SCALE = False  # Disabled due to performance impact

        # --- Database Connection ---
        try:
            # Establish connection and check server status
            self.mongo_client = pymongo.MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=5000)
            self.mongo_client.server_info()
            self.db = self.mongo_client[MONGO_DB_NAME]
            self.persons_collection = self.db[MONGO_COLLECTION_NAME]
            self.faiss_collection = self.db[MONGO_FAISS_COLLECTION_NAME]
            # Ensure unique index on permanent_id for data integrity
            self.persons_collection.create_index("permanent_id", unique=True)
            print("Successfully connected to MongoDB.")
        except pymongo.errors.ServerSelectionTimeoutError as e:
            print(f"!!!! MongoDB CONNECTION ERROR !!!!: {e}")
            exit()

        # --- Load Models ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading Object Detector (YOLOv8)...")
        self.detector = YOLO('yolov8m.pt')
        self.detector.to(self.device)

        print("Loading Re-ID Model (OSNet)...")
        # BoostTrack handles the internal loading of the Re-ID model
        self.tracker = BoostTrack(reid_weights=Path('osnet_x1_0_msmt17.pt'),
                                  device=(0 if self.device.type == 'cuda' else 'cpu'),
                                  half=False, with_reid=True)
        if not hasattr(self.tracker, 'reid_model') or self.tracker.reid_model is None:
            raise RuntimeError(f"Re-ID model failed to load! Exiting.")
        else:
            print("SUCCESS: Re-ID MODEL LOADED.")

        # --- Load Data ---
        self.load_db()

    def rebuild_faiss_index(self):
        """
        Builds a new HNSW Faiss index from all master vectors in the local cache.
        It uses an atomic swap mechanism (with a lock) to minimize disruption
        to ongoing search operations during the rebuild process.
        The resulting index is persisted to MongoDB for cold start readiness.
        """
        print("Building new HNSW Faiss index...")

        # 1. Collect data (vectors and P-IDs)
        all_vectors = []
        temp_pid_map = []

        # Create snapshot of the DB cache to avoid locking the entire time
        db_snapshot = dict(self.known_persons_db)

        sorted_pids = sorted(db_snapshot.keys())
        for pid in sorted_pids:
            data = db_snapshot[pid]
            # Use only the single master_vector per person
            all_vectors.append(data['master_vector'])
            temp_pid_map.append(pid)

        # 2. Create and configure the new index (WITHOUT lock)
        # HNSWFlat uses Inner Product for similarity (dot product for L2 normalized vectors)
        new_index = faiss.IndexHNSWFlat(VECTOR_DIMENSION, 48, faiss.METRIC_INNER_PRODUCT)
        new_index.hnsw.efConstruction = 200  # Higher quality during build
        new_index.hnsw.efSearch = 64  # Search quality

        if not all_vectors:
            print("No vectors to index. HNSW index initialized empty.")
            # Acquire lock for the atomic swap
            with self.faiss_lock:
                self.faiss_index = new_index
                self.index_to_pid_map = []

            # Clean up MongoDB
            try:
                self.faiss_collection.delete_one({"name": FAISS_INDEX_KEY})
            except Exception as e:
                print(f"!!!! FAISS (empty) index delete error !!!!: {e}")
            return

        # 3. Train and populate the index (WITHOUT lock)
        vector_matrix = np.stack(all_vectors, axis=0).astype('float32')
        # Training is usually not strictly necessary for HNSWFlat, but good practice
        new_index.train(vector_matrix)
        new_index.add(vector_matrix)

        print(f"New HNSW index built with {new_index.ntotal} vectors. Preparing atomic swap...")

        # 4. ATOMIC SWAP (The critical section, very brief)
        with self.faiss_lock:
            # Store reference to old index (for garbage collection)
            old_index = self.faiss_index
            self.faiss_index = new_index
            self.index_to_pid_map = temp_pid_map
            print(f"✓ Index swapped atomically. Active vectors: {self.faiss_index.ntotal}")

        # 5. Persist to MongoDB (outside the lock, in background)
        try:
            faiss.write_index(new_index, TEMP_FAISS_FILENAME)
            with open(TEMP_FAISS_FILENAME, "rb") as f:
                index_buffer = f.read()
            self.faiss_collection.update_one(
                {"name": FAISS_INDEX_KEY},
                {"$set": {"index_data": Binary(index_buffer), "last_updated": datetime.now()}},
                upsert=True  # Insert if not exists, update otherwise
            )
            print(f"✓ HNSW Faiss index persisted to MongoDB.")
        except Exception as e:
            print(f"!!!! FAISS SAVE ERROR (non-critical) !!!!: {e}")
        finally:
            if os.path.exists(TEMP_FAISS_FILENAME):
                os.remove(TEMP_FAISS_FILENAME)

        self.index_build_in_progress = False

    def calculate_sharpness(self, image):
        """
        Calculates the sharpness of an image using the variance of the Laplacian.
        Higher variance indicates more details and better focus (less blur).

        Args:
            image (np.ndarray): The BGR image crop.

        Returns:
            float: The Laplacian variance (sharpness score).
        """
        if image is None or image.size == 0:
            return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

    def advanced_quality_check(self, crop, bbox, frame_shape):
        """
        Performs a multi-criteria quality assessment on a detected person crop
        to decide if it's suitable for enrollment into the Re-ID database.

        Criteria include: Aspect Ratio, Bounding Box Area, Sharpness, Edge Density,
        Brightness Variance, and Position relative to frame edges.

        Args:
            crop (np.ndarray): The image region of the detected person.
            bbox (list): The bounding box coordinates [x1, y1, x2, y2].
            frame_shape (tuple): The shape of the original frame (H, W, C).

        Returns:
            tuple: (is_good_quality: bool, reason: str, quality_score: float)
        """
        h, w = crop.shape[:2]
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        quality_scores = {}
        rejection_reasons = []

        # 1. ASPECT RATIO CHECK (person should be ~2:1 height/width)
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 0.5:
            rejection_reasons.append("aspect_too_wide")
            quality_scores['aspect'] = 0.0
        elif aspect_ratio > 4.0:
            rejection_reasons.append("aspect_too_tall")
            quality_scores['aspect'] = 0.0
        else:
            if 1.5 <= aspect_ratio <= 3.0:
                quality_scores['aspect'] = 1.0
            else:
                quality_scores['aspect'] = 0.5

        # 2. BOUNDING BOX AREA
        bbox_area = h * w
        if bbox_area < MIN_BBOX_AREA_ENROLL:
            rejection_reasons.append("too_small")
            quality_scores['size'] = 0.0
        else:
            # Normalized score based on area
            quality_scores['size'] = min(1.0, bbox_area / 50000.0)

        # 3. SHARPNESS (Laplacian variance)
        sharpness = self.calculate_sharpness(crop)
        if sharpness < MIN_SHARPNESS_ENROLL:
            rejection_reasons.append("too_blurry")
            quality_scores['sharpness'] = 0.0
        else:
            # Normalized score based on sharpness
            quality_scores['sharpness'] = min(1.0, sharpness / 200.0)

        # 4. EDGE DENSITY (Indicator for heavy occlusion or clutter)
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)

            if edge_density < 0.02:  # Too smooth
                rejection_reasons.append("occluded")
                quality_scores['edges'] = 0.0
            elif edge_density > 0.30:  # Too busy
                quality_scores['edges'] = 0.5
            else:
                quality_scores['edges'] = 1.0 if 0.05 <= edge_density <= 0.20 else 0.7
        except Exception as e:
            quality_scores['edges'] = 0.5

        # 5. BRIGHTNESS VARIANCE (Illumination check)
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            brightness_std = gray.std()
            brightness_mean = gray.mean()

            if brightness_std < 15:
                rejection_reasons.append("poor_illumination")
                quality_scores['brightness'] = 0.0
            elif brightness_mean < 30 or brightness_mean > 225:
                rejection_reasons.append("extreme_lighting")
                quality_scores['brightness'] = 0.3
            else:
                quality_scores['brightness'] = 1.0
        except Exception as e:
            quality_scores['brightness'] = 0.5

        # 6. CENTER POSITION (Avoid truncated detections at frame edges)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        margin_x = min(center_x, frame_w - center_x) / frame_w
        margin_y = min(center_y, frame_h - center_y) / frame_h

        if margin_x < 0.05 or margin_y < 0.05:  # Within 5% margin
            rejection_reasons.append("at_frame_edge")
            quality_scores['position'] = 0.0
        else:
            quality_scores['position'] = 1.0

        # AGGREGATE SCORE (Weighted average)
        weights = {
            'aspect': 1.5,
            'size': 1.5,
            'sharpness': 2.0,
            'edges': 1.0,
            'brightness': 1.0,
            'position': 0.5
        }

        total_score = sum(quality_scores[k] * weights[k] for k in quality_scores.keys())
        max_score = sum(weights.values())
        normalized_score = total_score / max_score

        # DECISION: Require minimum 0.6 overall quality and no hard rejections
        is_good_quality = (normalized_score >= 0.6) and (len(rejection_reasons) == 0)

        # Format reason
        if rejection_reasons:
            reason = ", ".join(rejection_reasons)
        else:
            reason = "good"

        return is_good_quality, reason, normalized_score

    def load_db(self):
        """
        Loads all person vectors from MongoDB into the local cache (`known_persons_db`).
        Attempts to load the Faiss index from MongoDB; if not found, is outdated,
        or fails to load, it triggers a rebuild of the index from the loaded vectors.
        """
        print("Loading database from MongoDB...")
        try:
            all_persons = list(self.persons_collection.find({}))
            if all_persons:
                for doc in all_persons:
                    pid = doc['permanent_id']
                    self.known_persons_db[pid] = {
                        "master_vector": np.frombuffer(doc['master_vector'], dtype=np.float32),
                        "count": doc['count'],
                        "created": doc['created']
                    }
                self.next_permanent_id = max(self.known_persons_db.keys()) + 1 if self.known_persons_db else 1
                print(
                    f"Successfully loaded {len(self.known_persons_db)} persons from MongoDB. Next ID: {self.next_permanent_id}")
            else:
                print("No person records found in MongoDB. Starting fresh.")
                self.next_permanent_id = 1

            # Faiss index loading logic
            index_doc = self.faiss_collection.find_one({"name": FAISS_INDEX_KEY})
            if index_doc and 'index_data' in index_doc:
                print("Loading Faiss index from MongoDB...")
                try:
                    index_buffer = index_doc['index_data']
                    with open(TEMP_FAISS_FILENAME, "wb") as f:
                        f.write(index_buffer)
                    loaded_index = faiss.read_index(TEMP_FAISS_FILENAME)

                    if hasattr(loaded_index, 'hnsw'):
                        loaded_index.hnsw.efSearch = 64

                    # Check for index staleness (vector count mismatch)
                    if loaded_index.ntotal != len(self.known_persons_db):
                        print("Faiss index is stale (vector count mismatch). Rebuilding...")
                        raise ValueError("Index stale")

                    # Atomic swap with lock
                    with self.faiss_lock:
                        self.faiss_index = loaded_index
                        # Rebuild index-to-pid map based on the loaded DB cache (sorted by PID)
                        print("Rebuilding index-to-pid map from loaded DB...")
                        temp_pid_map = []
                        sorted_pids = sorted(self.known_persons_db.keys())
                        for pid in sorted_pids:
                            temp_pid_map.append(pid)
                        self.index_to_pid_map = temp_pid_map

                    print("Faiss index loaded successfully from MongoDB.")
                except Exception as e:
                    print(f"Failed to load Faiss index from MongoDB ({e}), rebuilding...")
                    self.rebuild_faiss_index()
                finally:
                    if os.path.exists(TEMP_FAISS_FILENAME):
                        os.remove(TEMP_FAISS_FILENAME)
            else:
                print("No Faiss index found in MongoDB. Building from scratch...")
                self.rebuild_faiss_index()
        except Exception as e:
            print(f"Error loading DB, attempting to create a new one: {e}")
            self.rebuild_faiss_index()
            self.known_persons_db.clear()
            self.next_permanent_id = 1

    def get_box_center(self, box):
        """
        Calculates the center (x, y) coordinates of a bounding box.

        Args:
            box (list): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            tuple: Center coordinates (center_x, center_y).
        """
        return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

    def calculate_iou(self, box1, box2):
        """
        Calculates the Intersection over Union (IoU) metric between two bounding boxes.

        Args:
            box1 (list): First bounding box [x1, y1, x2, y2].
            box2 (list): Second bounding box [x1, y1, x2, y2].

        Returns:
            float: IoU score (0.0 to 1.0).
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection area calculation
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Union area calculation
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou

    def predict_next_position(self, last_box, time_elapsed):
        """
        Simple prediction of a person's next position based on the last known box.
        This simplified version assumes minimal movement and uses box expansion
        to increase the search area based on time elapsed.

        Args:
            last_box (list): Last known bounding box [x1, y1, x2, y2].
            time_elapsed (float): Seconds since the person was last seen.

        Returns:
            list: Predicted bounding box [x1, y1, x2, y2] (expanded).
        """
        # Simple expansion based on time (capped at 100 pixels)
        expansion_factor = min(50 * time_elapsed, 100)

        x1, y1, x2, y2 = last_box
        predicted_box = [
            x1 - expansion_factor,
            y1 - expansion_factor,
            x2 + expansion_factor,
            y2 + expansion_factor
        ]

        return predicted_box

    def adaptive_spatial_threshold(self, bbox):
        """
        Calculates a dynamic distance threshold for spatial matching based on the
        size of the detected bounding box. Larger objects allow for a greater
        movement tolerance.

        Args:
            bbox (list): Bounding box [x1, y1, x2, y2].

        Returns:
            float: Adaptive distance threshold in pixels.
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_diag = np.sqrt(bbox_width ** 2 + bbox_height ** 2)

        # Threshold is proportional to diagonal size (0.5x, capped at 200, min 30)
        threshold = min(bbox_diag * 0.5, 200.0)
        threshold = max(threshold, 30.0)

        return threshold

    def l2_normalize(self, v):
        """
        Normalizes a vector to unit length (L2 norm). Essential for Inner Product
        search in Faiss to represent cosine similarity.

        Args:
            v (np.ndarray): The feature vector.

        Returns:
            np.ndarray: The L2 normalized vector.
        """
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    def enroll_person(self, vectors):
        """
        Enrolls a new person by calculating the master vector (mean of all collected vectors),
        assigning a new Permanent ID (P-ID), updating the local cache, and
        queuing the new document for asynchronous insertion into MongoDB.
        It also triggers a Faiss index rebuild if the batch size threshold is met.

        Args:
            vectors (list): List of feature vectors collected for the new person.

        Returns:
            int: The newly assigned Permanent ID (P-ID).
        """
        if not vectors:
            raise ValueError("No vectors provided to enroll.")

        master = self.l2_normalize(np.mean(np.stack(vectors, axis=0), axis=0))
        current_dt = datetime.now()
        new_pid = self.next_permanent_id

        new_person_doc = {
            "permanent_id": new_pid,
            "master_vector": Binary(master.tobytes()),  # Store as MongoDB Binary type
            "count": len(vectors),
            "created": current_dt,
            "last_seen": current_dt
        }

        # Queue for asynchronous database insert
        self.db_write_queue.put({
            "type": "insert",
            "doc": new_person_doc
        })
        print(f"MongoDB INSERT: New person (P{new_pid}) queued for async write.")

        # Add to local cache
        self.known_persons_db[new_pid] = {
            "master_vector": master,
            "count": len(vectors),
            "created": current_dt
        }

        print(f"CACHE ADD: P{new_pid} added to cache.")
        self.next_permanent_id += 1

        # Add to pending enrollments for batch rebuild
        self.pending_enrollments.append(new_pid)

        # Check if rebuild should be triggered
        time_since_rebuild = time.time() - self.last_index_rebuild
        should_rebuild = (
                len(self.pending_enrollments) >= self.BATCH_SIZE_TRIGGER or
                time_since_rebuild > self.MIN_REBUILD_INTERVAL
        )

        if should_rebuild and not self.index_build_in_progress:
            print(f"Triggering index rebuild (pending: {len(self.pending_enrollments)})")
            self.index_build_in_progress = True
            self.pending_enrollments.clear()
            self.last_index_rebuild = time.time()

            # Start rebuild in a separate thread (non-blocking)
            rebuild_thread = threading.Thread(target=self.rebuild_faiss_index, daemon=True)
            rebuild_thread.start()

        return new_pid

    def cleanup_stale_buffers(self, current_time):
        """
        Performs periodic cleanup of buffers (`recent_vector_buffers`, `last_update_time_for_pid`)
        to remove entries related to P-IDs that no longer exist in the database or have been
        inactive for an excessively long period, preventing memory leaks.

        Args:
            current_time (float): The current time stamp.
        """
        stale_pids = []

        # 1. Check for missing P-IDs in the main database cache
        for pid in list(self.recent_vector_buffers.keys()):
            if pid not in self.known_persons_db:
                stale_pids.append(pid)
            elif (current_time - self.last_update_time_for_pid.get(pid, 0)) > self.BUFFER_MAX_IDLE_TIME:
                stale_pids.append(pid)

        # 2. Check last_update_time_for_pid against the main DB cache
        for pid in list(self.last_update_time_for_pid.keys()):
            if pid not in self.known_persons_db:
                if pid not in stale_pids:
                    stale_pids.append(pid)

        # 3. Remove all stale entries from various buffers
        if stale_pids:
            for pid in stale_pids:
                if pid in self.recent_vector_buffers:
                    del self.recent_vector_buffers[pid]
                if pid in self.last_update_time_for_pid:
                    del self.last_update_time_for_pid[pid]
                if pid in self.last_known_box_for_pid:
                    del self.last_known_box_for_pid[pid]

            print(f"BUFFER CLEANUP: Removed {len(stale_pids)} stale buffer(s). "
                  f"Active buffers: {len(self.recent_vector_buffers)}")

    def incremental_update(self, perm_id, new_vector):
        """
        Updates the master vector for an enrolled person in the local cache using an
        Exponential Moving Average (EMA) and queues the update for asynchronous
        persistence in MongoDB, provided the vector update limit is not yet reached.

        Args:
            perm_id (int): The Permanent ID (P-ID) of the person.
            new_vector (np.ndarray): The feature vector from the current frame.
        """
        entry = self.known_persons_db.get(perm_id)
        if entry is None:
            return

        n = entry.get("count", 1)
        mongo_update_doc = {"$set": {"last_seen": datetime.now()}}

        # Add vector to the recent buffer
        buffer = self.recent_vector_buffers[perm_id]
        buffer.append(new_vector)

        # Update master vector if under the limit
        if n < MAX_VECTORS_FOR_UPDATE:
            master = entry["master_vector"]

            # EMA calculation
            alpha = 0.1
            updated_master = self.l2_normalize((1 - alpha) * master + alpha * new_vector)

            new_count = n + 1
            mongo_update_doc["$set"]["master_vector"] = Binary(updated_master.tobytes())
            mongo_update_doc["$set"]["count"] = new_count

            entry["master_vector"] = updated_master
            entry["count"] = new_count

            if new_count == MAX_VECTORS_FOR_UPDATE:
                print(f"INFO: Person P{perm_id} vector locked after {new_count} updates.")

        # Queue for asynchronous database update
        self.db_write_queue.put({
            "type": "update",
            "pid": perm_id,
            "update_doc": mongo_update_doc
        })
        # Note: No print statement here to prevent console spam due to high frequency

    def find_best_match(self, query_vector):
        """
        Performs a fast, thread-safe similarity search against the HNSW Faiss index
        to find the closest match for the query vector among enrolled persons.

        Args:
            query_vector (np.ndarray): The feature vector to query.

        Returns:
            tuple: (matched_pid: int | None, best_similarity: float)
        """
        with self.faiss_lock:
            if self.faiss_index is None or self.faiss_index.ntotal == 0 or not self.index_to_pid_map:
                return None, -1.0

            try:
                # Faiss expects a 2D float32 numpy array
                query_vector_2d = query_vector.reshape(1, -1).astype('float32')
                # Search for the 1 nearest neighbor
                D, I = self.faiss_index.search(query_vector_2d, k=1)
                best_similarity = D[0][0]

                if best_similarity >= self.current_match_threshold:
                    matched_index_position = I[0][0]
                    matched_pid = self.index_to_pid_map[matched_index_position]
                    return matched_pid, best_similarity
            except Exception as e:
                print(f"!!!! FAISS (HNSW) SEARCH ERROR !!!!: {e}")

        return None, -1.0

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
            return [self.l2_normalize(np.random.randn(VECTOR_DIMENSION).astype(np.float32)) for _ in crops]

        # Filter and process valid crops
        valid_crops = [c for c in crops if c.size > 0]
        if not valid_crops:
            return []

        # Resize and apply advanced preprocessing
        resized_crops = [
            cv2.resize(
                self.preprocess_crop_advanced(c),
                REID_INPUT_SIZE,
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

    def detector_worker(self):
        """
        Runs the YOLOv8 object detector in a separate, non-blocking thread.
        It continuously pulls frames from the input queue, performs detection
        (only for the 'person' class), filters results based on the live threshold,
        and pushes the detections to the output queue.
        """
        print("Detector thread (YOLOv8) started...")
        while True:
            frame_to_process = self.frame_queue.get()
            if frame_to_process is None:  # Exit signal
                print("Detector thread stopping...")
                break

            # Run YOLO detection for 'person' class (class_id=0)
            output = self.detector(frame_to_process, classes=0, verbose=False)

            if not output:
                new_detections = np.empty((0, 6), dtype=np.float32)
            else:
                boxes = output[0].boxes

                # Filter detections based on the current live-tuning threshold
                scores = boxes.conf.cpu().numpy()
                keep_mask = (scores >= self.current_detection_threshold)

                if keep_mask.any():
                    # Format as BoostTrack expects: [x1, y1, x2, y2, score, class_id]
                    new_detections = boxes.data[keep_mask].cpu().numpy()

                    # Safety check: limit columns to 6 (may include tracker_id if BoostTrack was used)
                    if new_detections.shape[1] > 6:
                        new_detections = new_detections[:, :6]
                else:
                    new_detections = np.empty((0, 6), dtype=np.float32)

            # Atomically put new detections into the queue
            try:
                self.detection_queue.get_nowait()
            except queue.Empty:
                pass
            self.detection_queue.put(new_detections)

    def run(self):
        """
        The main application loop. It handles video capture, tracking update,
        batch feature extraction, Re-ID matching, spatial re-acquisition,
        enrollment logic, drawing results, and user input processing.
        """
        detector_thread = threading.Thread(target=self.detector_worker, daemon=True)
        detector_thread.start()

        cap = cv2.VideoCapture("7.mp4")  # Assumes default camera
        if not cap.isOpened():
            print("!!!! ERROR: Could not open camera. !!!!")
            return

        WINDOW_NAME = "ReID Vectors (MongoDB - Faiss)"
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
                if frame_count % DETECTION_INTERVAL == 0:
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
                    x1 = max(0, min(w - 1, x1));
                    x2 = max(0, min(w - 1, x2))
                    y1 = max(0, min(h - 1, y1));
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

                    if track_id in self.track_id_to_permanent_id_map:
                        # Existing known person
                        pid = self.track_id_to_permanent_id_map[track_id]
                        self.last_known_box_for_pid[pid] = box
                        label, color = f"P{pid}", (0, 255, 0)  # Green for P-ID match

                        # Incremental master vector update check
                        if (time.time() - self.last_update_time_for_pid[pid]) > self.UPDATE_INTERVAL:
                            self.incremental_update(pid, current_vector)
                            self.last_update_time_for_pid[pid] = time.time()
                    else:
                        # --- ADVANCED Spatial Re-acquisition with Motion Prediction ---
                        now = time.time()
                        new_box_center = self.get_box_center(box)
                        spatial_match_pid = None
                        best_match_score = -1.0

                        for pid_to_check, data in self.lost_pid_buffer.items():
                            time_elapsed = now - data["disappear_time"]
                            last_box = data["last_box"]

                            # Predict position and calculate spatial metrics
                            predicted_box = self.predict_next_position(last_box, time_elapsed)
                            iou = self.calculate_iou(box, predicted_box)
                            lost_box_center = self.get_box_center(last_box)
                            center_dist = np.linalg.norm(np.array(new_box_center) - np.array(lost_box_center))
                            adaptive_threshold = self.adaptive_spatial_threshold(box)

                            # Time decay factor (recent loss is more reliable)
                            time_factor = np.exp(-time_elapsed / SHORT_TERM_MEMORY_TTL)

                            # Weighted combination of spatial metrics
                            iou_score = iou * 2.0
                            dist_score = max(0, 1.0 - (center_dist / adaptive_threshold))
                            time_score = time_factor

                            match_score = (iou_score * 0.5 + dist_score * 0.3 + time_score * 0.2)

                            if match_score > 0.3 and match_score > best_match_score:
                                best_match_score = match_score
                                spatial_match_pid = pid_to_check

                        if spatial_match_pid is not None:
                            # SPATIAL MATCH FOUND: Assign P-ID to the current T-ID
                            self.track_id_to_permanent_id_map[track_id] = spatial_match_pid
                            label = f"P{spatial_match_pid} (ReAcq:{best_match_score:.2f})"
                            color = (0, 255, 128)  # Greenish for re-acquired

                            # Remove from lost buffer and update position
                            del self.lost_pid_buffer[spatial_match_pid]
                            self.last_known_box_for_pid[spatial_match_pid] = box
                            print(f"DEBUG: P{spatial_match_pid} spatially re-acquired by T{track_id}. "
                                  f"Match score: {best_match_score:.2f}")
                        else:
                            # --- Re-ID Match Search (Faiss) ---
                            match_id, match_sim = self.find_best_match(current_vector)

                            if match_id is not None:
                                # Re-ID match found
                                self.track_id_to_permanent_id_map[track_id] = match_id
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

                                buf = self.active_track_buffers[track_id]
                                if is_good_quality:
                                    # Collect vector if quality is sufficient
                                    buf["vectors"].append(current_vector)
                                    buf["quality_scores"] = buf.get("quality_scores", [])
                                    buf["quality_scores"].append(quality_score)

                                    # Display enrollment progress and quality
                                    avg_quality = np.mean(buf["quality_scores"])
                                    label = f"New? T{track_id} ({len(buf['vectors'])}/{MIN_ENROLL_FRAMES}) Q:{avg_quality:.2f}"
                                else:
                                    label = f"New? T{track_id} (REJECT: {reason})"

                                buf["last_seen"] = time.time()

                                # Enrollment decision
                                if len(buf["vectors"]) >= MIN_ENROLL_FRAMES:
                                    avg_quality = np.mean(buf.get("quality_scores", [0.5]))

                                    # Final quality threshold check
                                    if avg_quality >= 0.65:
                                        new_pid = self.enroll_person(buf["vectors"])
                                        if new_pid:
                                            print(f"ENROLLED: New person P{new_pid}. "
                                                  f"Vectors: {len(buf['vectors'])} | Avg Quality: {avg_quality:.2f}")
                                            self.track_id_to_permanent_id_map[track_id] = new_pid
                                            del self.active_track_buffers[track_id]
                                    else:
                                        print(f"REJECTED ENROLLMENT: T{track_id} failed quality threshold. "
                                              f"Avg Quality: {avg_quality:.2f} < 0.65. Resetting buffer.")
                                        # Reset buffer to collect better samples
                                        buf["vectors"].clear()
                                        buf["quality_scores"].clear()

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- 5. Cleanup ---
                now = time.time()

                # 1. Clean up stale active track buffers (T-IDs)
                stale_active_tracks = [tid for tid, b in self.active_track_buffers.items() if
                                       (now - b.get("last_seen", now)) > 5.0]
                for tid in stale_active_tracks:
                    del self.active_track_buffers[tid]

                # 2. Identify and process disappeared P-IDs
                disappeared_track_ids = [tid for tid in self.track_id_to_permanent_id_map if
                                         tid not in current_frame_track_ids]

                for tid in disappeared_track_ids:
                    pid = self.track_id_to_permanent_id_map.pop(tid, None)
                    if pid:
                        # Add to 'lost' buffer for short-term re-acquisition
                        if pid in self.last_known_box_for_pid:
                            last_box = self.last_known_box_for_pid[pid]
                            velocity = [0, 0]  # Placeholder for future Kalman/BoostTrack velocity

                            # Update lost buffer with disappear time and last box
                            self.lost_pid_buffer[pid] = {
                                "disappear_time": now,
                                "last_box": last_box,
                                "velocity": velocity
                            }
                        else:
                            # If no box was saved (rare case), clean up associated buffers
                            if pid in self.recent_vector_buffers:
                                del self.recent_vector_buffers[pid]

                # 3. Clean up P-IDs that have expired from short-term memory (TTL)
                stale_lost_pids = [pid for pid, data in self.lost_pid_buffer.items() if
                                   (now - data["disappear_time"]) > SHORT_TERM_MEMORY_TTL]

                for pid in stale_lost_pids:
                    del self.lost_pid_buffer[pid]

                    # Also clean up long-term buffers associated with the fully lost PID
                    if pid in self.recent_vector_buffers:
                        del self.recent_vector_buffers[pid]
                    if pid in self.last_known_box_for_pid:
                        del self.last_known_box_for_pid[pid]

                # 4. Periodic long-term buffer cleanup
                if (now - self.last_buffer_cleanup) > self.BUFFER_CLEANUP_INTERVAL:
                    self.cleanup_stale_buffers(now)
                    self.last_buffer_cleanup = now

                # Display HUD (Head-Up Display) information
                tta_indicator = "TTA:ON" if self.USE_TTA else "TTA:OFF"

                hud_text1 = f"FPS: {fps:.1f} | Detection (U/J): {self.current_detection_threshold:.2f}"
                hud_text2 = f"Match (Y/H): {self.current_match_threshold:.2f} | DB: {len(self.known_persons_db)} | {tta_indicator}"

                # Draw background rectangle for readability
                cv2.rectangle(frame, (5, 5), (450, 60), (0, 0, 0), -1)

                # Draw the text
                cv2.putText(frame, hud_text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, hud_text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
            self.frame_queue.put(None)  # Signal detector thread to stop
            return 'q'

        # Detection Threshold (U=Up, J=Down)
        elif key == ord('u'):
            self.current_detection_threshold = min(1.0, self.current_detection_threshold + 0.05)
            print(f"Detection Threshold set to: {self.current_detection_threshold:.2f}")
        elif key == ord('j'):
            self.current_detection_threshold = max(0.0, self.current_detection_threshold - 0.05)
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

        # Reset Database and State (R)
        elif key == ord('r'):
            print("RESET: Queuing MongoDB clear operation...")
            self.db_write_queue.put({"type": "reset"})  # Async database reset

            # Clear local state
            self.known_persons_db.clear()
            self.next_permanent_id = 1
            self.active_track_buffers.clear()
            self.track_id_to_permanent_id_map.clear()
            self.lost_pid_buffer.clear()
            self.last_known_box_for_pid.clear()
            self.recent_vector_buffers.clear()
            self.last_update_time_for_pid.clear()
            self.rebuild_faiss_index()  # Rebuild empty index
            print("Program memory (cache and index) has been cleared.")

        # Debug Status (D)
        elif key == ord('d'):
            with self.faiss_lock:
                print("\n--- DB STATUS (d) ---")
                print(f"Persons in cache: {len(self.known_persons_db)}")
                print(f"Vectors in Faiss index: {self.faiss_index.ntotal if self.faiss_index else 0}")
                print(f"Next ID: {self.next_permanent_id}")
                for pid, data in self.known_persons_db.items():
                    created_date = data.get('created', 'N/A')
                    if isinstance(created_date, datetime):
                        created_date = created_date.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"  - P{pid}: {data['count']} vectors, Created: {created_date}")
                print("------------------\n")


        # Delete Last Person (K)
        elif key == ord('k'):
            if self.known_persons_db:
                try:
                    last_id = max(self.known_persons_db.keys())

                    # Queue async delete and clear local state
                    self.db_write_queue.put({"type": "delete", "pid": last_id})
                    print(f"DELETED (k): P{last_id} delete queued.")

                    del self.known_persons_db[last_id]
                    if last_id in self.recent_vector_buffers: del self.recent_vector_buffers[last_id]
                    if last_id in self.last_update_time_for_pid: del self.last_update_time_for_pid[last_id]
                    if last_id in self.last_known_box_for_pid: del self.last_known_box_for_pid[last_id]

                    self.active_track_buffers.clear()
                    self.track_id_to_permanent_id_map.clear()
                    self.lost_pid_buffer.clear()

                    self.rebuild_faiss_index()
                    print(f"DELETED (k): P{last_id} deleted from local cache. Index rebuilt.")
                except Exception as e:
                    print(f"Error deleting person: {e}")
            else:
                print("DB is already empty. Nothing to delete.")
        return key

    def cleanup(self):
        """
        Releases all system resources gracefully upon application shutdown.
        This includes stopping worker threads, closing OpenCV windows, and
        closing the MongoDB connection.
        """
        self.stop_event.set()

        print("Cleaning up and exiting...")

        # Gracefully shut down async DB writer
        print("Waiting for pending database writes to complete...")
        self.db_write_queue.put(None)  # Send poison pill
        self.db_writer_thread.join(timeout=5.0)  # Wait for thread to finish

        if self.db_writer_thread.is_alive():
            print("WARNING: Database writer thread did not exit cleanly.")
        else:
            print("Database writer thread exited cleanly.")

        cv2.destroyAllWindows()
        self.mongo_client.close()
        print(f"Exited cleanly. DB entries in cache: {len(self.known_persons_db)}")

    def _db_writer_worker(self):
        """
        Background thread that processes database write and delete operations asynchronously
        from the queue. This prevents the main video processing loop from being blocked
        by slow MongoDB I/O, maintaining real-time performance.
        """
        print("Database writer thread started...")

        while not self.stop_event.is_set():
            try:
                # Get operation with a timeout to allow stop_event check
                operation = self.db_write_queue.get(timeout=1.0)

                if operation is None:  # Poison pill (shutdown signal)
                    break

                op_type = operation.get("type")

                # Process INSERT operation
                if op_type == "insert":
                    doc = operation["doc"]
                    try:
                        self.persons_collection.insert_one(doc)
                        pid = doc["permanent_id"]
                        print(f"[ASYNC] MongoDB INSERT: P{pid} completed.")
                    except Exception as e:
                        print(f"!!!! [ASYNC] MongoDB INSERT ERROR (P{pid}) !!!!: {e}")

                # Process UPDATE operation
                elif op_type == "update":
                    pid = operation["pid"]
                    update_doc = operation["update_doc"]
                    try:
                        self.persons_collection.update_one(
                            {"permanent_id": int(pid)},
                            update_doc
                        )
                        # Silent update log to prevent console spam
                    except Exception as e:
                        print(f"!!!! [ASYNC] MongoDB UPDATE ERROR (P{pid}) !!!!: {e}")

                # Process DELETE operation
                elif op_type == "delete":
                    pid = operation["pid"]
                    try:
                        self.persons_collection.delete_one({"permanent_id": int(pid)})
                        print(f"[ASYNC] MongoDB DELETE: P{pid} completed.")
                    except Exception as e:
                        print(f"!!!! [ASYNC] MongoDB DELETE ERROR (P{pid}) !!!!: {e}")

                # Process RESET operation
                elif op_type == "reset":
                    try:
                        self.persons_collection.delete_many({})
                        self.faiss_collection.delete_many({})
                        print("[ASYNC] MongoDB RESET completed (all data cleared).")
                    except Exception as e:
                        print(f"!!!! [ASYNC] MongoDB RESET ERROR !!!!: {e}")

                else:
                    print(f"!!!! Unknown DB operation type: {op_type} !!!!")

            except queue.Empty:
                # Timeout occurred, loop continues to check stop_event
                continue
            except Exception as e:
                print(f"!!!! [ASYNC] DB writer unexpected error !!!!: {e}")

        print("Database writer thread exited.")


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