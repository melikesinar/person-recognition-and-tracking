# main_reid_vectors.py
import os
import cv2
import time
import queue
import torch
import pymongo
import threading
import numpy as np
import faiss  # type: ignore
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
MONGO_CONNECTION_STRING = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("COLLECTION")
MONGO_FAISS_COLLECTION_NAME = os.getenv("FAISS_COLLECTION")  # New collection to store the Faiss index
FAISS_INDEX_KEY = "primary_person_index"  # Key to store the index in MongoDB
TEMP_FAISS_FILENAME = "temp_faiss.index"  # Filename for temporarily writing the index to disk

# We load them here (outside the class) because detector_worker needs them
# We set defaults in case they are missing from .env
DEFAULT_DETECTION_SCORE = 0.7
DEFAULT_MATCH_THRESHOLD = 0.85

DETECTION_SCORE_THRESHOLD = float(os.getenv("DETECTION_SCORE_THRESHOLD", DEFAULT_DETECTION_SCORE))
MATCH_SIMILARITY_THRESHOLD = float(os.getenv("MATCH_SIMILARITY_THRESHOLD", DEFAULT_MATCH_THRESHOLD))
MIN_ENROLL_FRAMES = int(os.getenv("MIN_ENROLL_FRAMES", "30"))
MIN_BBOX_AREA_ENROLL = int(os.getenv("MIN_BBOX_AREA_ENROLL", "10000"))
MIN_SHARPNESS_ENROLL = float(os.getenv("MIN_SHARPNESS_ENROLL", "50.0"))
MAX_VECTORS_FOR_UPDATE = int(os.getenv("MAX_VECTORS_FOR_UPDATE", "300"))
RECENT_VECTOR_BUFFER_SIZE = int(os.getenv("RECENT_VECTOR_BUFFER_SIZE", "50"))
DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL", "2"))
REID_INPUT_SIZE = (128, 256)
VECTOR_DIMENSION = 512
SHORT_TERM_MEMORY_TTL = 2.0
SPATIAL_MATCH_THRESHOLD = 150


class PersonTrackerApp:
    """
    Main class that encapsulates all logic for the person Re-ID and tracking system.
    """

    def __init__(self):
        """
        Initializes the class, connects to the database, loads models, and prepares the state.
        """
        print("System initializing...")

        # --- State Variables ---
        self.faiss_index = None
        self.faiss_lock = threading.Lock()
        self.known_persons_db = {}
        self.index_to_pid_map = []
        self.lost_pid_buffer = {}
        self.last_known_box_for_pid = {}
        self.next_permanent_id = 1
        self.index_build_in_progress = False
        self.pending_enrollments = []  # Yeni enrollment'ları topla
        self.last_index_rebuild = time.time()
        self.MIN_REBUILD_INTERVAL = 30  # En az 30 saniyede bir rebuild
        self.BATCH_SIZE_TRIGGER = 5  # 5 yeni kişi eklenince rebuild

        self.stop_event = threading.Event()

        # These will be modified by keyboard presses
        self.current_match_threshold = MATCH_SIMILARITY_THRESHOLD
        self.current_detection_threshold = DETECTION_SCORE_THRESHOLD

        # --- Thread Communication Queues ---
        self.frame_queue = queue.Queue(maxsize=1)
        self.detection_queue = queue.Queue(maxsize=1)

        # --- Tracking and Enrollment Buffers ---
        self.active_track_buffers = defaultdict(lambda: {
            "vectors": [],
            "crops": [],
            "quality_scores": [],
            "last_seen": time.time()
        })
        self.track_id_to_permanent_id_map = {}
        self.recent_vector_buffers = defaultdict(lambda: deque(maxlen=RECENT_VECTOR_BUFFER_SIZE))
        self.last_update_time_for_pid = defaultdict(float)
        self.UPDATE_INTERVAL = 1.0  # 1 second
        self.BUFFER_CLEANUP_INTERVAL = 60.0  # Her 60 saniyede cleanup
        self.BUFFER_MAX_IDLE_TIME = 3600.0  # 1 saat görunmeyen bufferi sil
        self.last_buffer_cleanup = time.time()

        # --- Database Connection ---
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=5000)
            self.mongo_client.server_info()
            self.db = self.mongo_client[MONGO_DB_NAME]
            self.persons_collection = self.db[MONGO_COLLECTION_NAME]
            self.faiss_collection = self.db[MONGO_FAISS_COLLECTION_NAME]
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

        print("Loading Re-ID Model...")
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
        Builds a new HNSW Faiss index WITHOUT locking.
        Uses atomic swap for minimal disruption.
        """
        print("Building new HNSW Faiss index (non-blocking)...")

        # 1. Veriyi topla (lock ALMADAN)
        all_vectors = []
        temp_pid_map = []

        # known_persons_db'yi kopyala (thread-safe)
        db_snapshot = dict(self.known_persons_db)  # Shallow copy yeterli

        sorted_pids = sorted(db_snapshot.keys())
        for pid in sorted_pids:
            data = db_snapshot[pid]
            # SADECE master_vector kullan (2x vector yaklaşımını bırakıyoruz)
            all_vectors.append(data['master_vector'])
            temp_pid_map.append(pid)

        # 2. Yeni index oluştur (lock OLMADAN)
        new_index = faiss.IndexHNSWFlat(VECTOR_DIMENSION, 48, faiss.METRIC_INNER_PRODUCT)
        new_index.hnsw.efConstruction = 200
        new_index.hnsw.efSearch = 64

        if not all_vectors:
            print("No vectors to index. HNSW index initialized empty.")
            # Atomic swap için lock AL
            with self.faiss_lock:
                self.faiss_index = new_index
                self.index_to_pid_map = []

            # MongoDB'den sil
            try:
                self.faiss_collection.delete_one({"name": FAISS_INDEX_KEY})
            except Exception as e:
                print(f"!!!! FAISS (empty) index delete error !!!!: {e}")
            return

        # 3. Index'i train ve populate et (lock OLMADAN)
        vector_matrix = np.stack(all_vectors, axis=0).astype('float32')
        new_index.train(vector_matrix)
        new_index.add(vector_matrix)

        print(f"New HNSW index built with {new_index.ntotal} vectors. Preparing atomic swap...")

        # 4. ATOMIC SWAP (kritik bölge, çok kısa sürecek)
        with self.faiss_lock:
            old_index = self.faiss_index  # Referansı sakla (GC için)
            self.faiss_index = new_index
            self.index_to_pid_map = temp_pid_map
            print(f"✓ Index swapped atomically. Active vectors: {self.faiss_index.ntotal}")

        # 5. MongoDB'ye kaydet (lock OLMADAN, arka planda)
        try:
            faiss.write_index(new_index, TEMP_FAISS_FILENAME)
            with open(TEMP_FAISS_FILENAME, "rb") as f:
                index_buffer = f.read()
            self.faiss_collection.update_one(
                {"name": FAISS_INDEX_KEY},
                {"$set": {"index_data": Binary(index_buffer), "last_updated": datetime.now()}},
                upsert=True
            )
            print(f"✓ HNSW Faiss index persisted to MongoDB.")
        except Exception as e:
            print(f"!!!! FAISS SAVE ERROR (non-critical) !!!!: {e}")
        finally:
            if os.path.exists(TEMP_FAISS_FILENAME):
                os.remove(TEMP_FAISS_FILENAME)

        # Artık old_index garbage collected olacak
        self.index_build_in_progress = False

    def calculate_sharpness(self, image):
        """Calculates the sharpness of an image using Laplacian variance."""
        if image is None or image.size == 0:
            return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

    def advanced_quality_check(self, crop, bbox, frame_shape):
        """
        Advanced quality assessment for enrollment using multiple criteria.
        Returns: (is_good_quality: bool, reason: str, quality_score: float)
        """
        h, w = crop.shape[:2]
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        quality_scores = {}
        rejection_reasons = []

        # 1. ASPECT RATIO CHECK (person should be ~2:1 height/width)
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 1.2:  # Too wide (sitting/lying or partial detection)
            rejection_reasons.append("aspect_too_wide")
            quality_scores['aspect'] = 0.0
        elif aspect_ratio > 4.0:  # Too tall (only legs or unrealistic)
            rejection_reasons.append("aspect_too_tall")
            quality_scores['aspect'] = 0.0
        else:
            # Optimal range: 1.5 - 3.0 (standing person)
            if 1.5 <= aspect_ratio <= 3.0:
                quality_scores['aspect'] = 1.0
            else:
                quality_scores['aspect'] = 0.5  # Acceptable but not ideal

        # 2. BOUNDING BOX AREA
        bbox_area = h * w
        if bbox_area < MIN_BBOX_AREA_ENROLL:
            rejection_reasons.append("too_small")
            quality_scores['size'] = 0.0
        else:
            # Normalize score (10000 = min, 50000 = excellent)
            quality_scores['size'] = min(1.0, bbox_area / 50000.0)

        # 3. SHARPNESS (Laplacian variance)
        sharpness = self.calculate_sharpness(crop)
        if sharpness < MIN_SHARPNESS_ENROLL:
            rejection_reasons.append("too_blurry")
            quality_scores['sharpness'] = 0.0
        else:
            # Normalize (50 = min, 200 = excellent)
            quality_scores['sharpness'] = min(1.0, sharpness / 200.0)

        # 4. EDGE DENSITY (occlusion detection)
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)

            if edge_density < 0.02:  # Too smooth = heavily occluded or blurred
                rejection_reasons.append("occluded")
                quality_scores['edges'] = 0.0
            elif edge_density > 0.30:  # Too busy = cluttered background
                quality_scores['edges'] = 0.5
            else:
                # Optimal range: 0.05 - 0.20
                quality_scores['edges'] = 1.0 if 0.05 <= edge_density <= 0.20 else 0.7
        except Exception as e:
            quality_scores['edges'] = 0.5  # Neutral if error

        # 5. BRIGHTNESS VARIANCE (illumination check)
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            brightness_std = gray.std()
            brightness_mean = gray.mean()

            if brightness_std < 15:  # Too uniform (over/under exposed)
                rejection_reasons.append("poor_illumination")
                quality_scores['brightness'] = 0.0
            elif brightness_mean < 30 or brightness_mean > 225:  # Too dark or bright
                rejection_reasons.append("extreme_lighting")
                quality_scores['brightness'] = 0.3
            else:
                quality_scores['brightness'] = 1.0
        except Exception as e:
            quality_scores['brightness'] = 0.5

        # 6. CENTER POSITION (avoid truncated detections at frame edges)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Calculate distance from frame edges (normalized)
        margin_x = min(center_x, frame_w - center_x) / frame_w
        margin_y = min(center_y, frame_h - center_y) / frame_h

        if margin_x < 0.05 or margin_y < 0.05:  # Too close to edge (within 5%)
            rejection_reasons.append("at_frame_edge")
            quality_scores['position'] = 0.0
        else:
            quality_scores['position'] = 1.0

        # AGGREGATE SCORE (weighted average)
        weights = {
            'aspect': 1.5,  # Very important (prevents bad crops)
            'size': 1.5,  # Very important
            'sharpness': 2.0,  # Most important (blurry = useless)
            'edges': 1.0,  # Important for occlusion
            'brightness': 1.0,  # Important for consistency
            'position': 0.5  # Less critical (can still be useful)
        }

        total_score = sum(quality_scores[k] * weights[k] for k in quality_scores.keys())
        max_score = sum(weights.values())
        normalized_score = total_score / max_score

        # DECISION: Require minimum 0.6 overall quality (60%)
        is_good_quality = (normalized_score >= 0.6) and (len(rejection_reasons) == 0)

        # Format reason
        if rejection_reasons:
            reason = ", ".join(rejection_reasons)
        else:
            reason = "good"

        return is_good_quality, reason, normalized_score

    def load_db(self):
        """
        Loads all person vectors from MongoDB into a local cache and
        loads the Faiss index from MongoDB (or rebuilds it if necessary).
        """
        print("Loading database from MongoDB...")
        try:
            all_persons = list(self.persons_collection.find({}))
            if all_persons:
                for doc in all_persons:
                    pid = doc['permanent_id']
                    self.known_persons_db[pid] = {
                        "master_vector": np.frombuffer(doc['master_vector'], dtype=np.float32),
                        # recent_master_vector ARTIK YOK
                        "count": doc['count'],
                        "created": doc['created']
                    }
                self.next_permanent_id = max(self.known_persons_db.keys()) + 1 if self.known_persons_db else 1
                print(
                    f"Successfully loaded {len(self.known_persons_db)} persons from MongoDB. Next ID: {self.next_permanent_id}")
            else:
                print("No person records found in MongoDB. Starting fresh.")
                self.next_permanent_id = 1

            # Faiss index loading aynı kalıyor (sadece ntotal check değişecek)
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

                    # ARTIK 1x vector per person
                    if loaded_index.ntotal != len(self.known_persons_db):
                        print("Faiss index is stale (vector count mismatch). Rebuilding...")
                        raise ValueError("Index stale")

                    with self.faiss_lock:
                        self.faiss_index = loaded_index
                        # Manuel haritayı yeniden oluştur
                        print("Rebuilding index-to-pid map from loaded DB...")
                        temp_pid_map = []
                        sorted_pids = sorted(self.known_persons_db.keys())
                        for pid in sorted_pids:
                            temp_pid_map.append(pid)  # Sadece master_vector
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
            print(f"Error loading DB, creating new one: {e}")
            self.rebuild_faiss_index()
            self.known_persons_db.clear()
            self.next_permanent_id = 1

    def get_box_center(self, box):
        """Calculates the center (x, y) of a bounding box."""
        return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

    def l2_normalize(self, v):
        """Normalizes a vector to unit length (L2 norm)."""
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    def enroll_person(self, vectors):
        """
        Enrolls a new person into the MongoDB and local cache.
        Triggers index rebuild if batch size reached.
        """
        if not vectors:
            raise ValueError("No vectors provided to enroll.")

        master = self.l2_normalize(np.mean(np.stack(vectors, axis=0), axis=0))
        current_dt = datetime.now()
        new_pid = self.next_permanent_id

        new_person_doc = {
            "permanent_id": new_pid,
            "master_vector": Binary(master.tobytes()),
            # recent_master_vector ARTIK YOK
            "count": len(vectors),
            "created": current_dt,
            "last_seen": current_dt
        }

        try:
            self.persons_collection.insert_one(new_person_doc)
            print(f"MongoDB INSERT: New person (P{new_pid}) added.")
        except Exception as e:
            print(f"!!!! MongoDB INSERT ERROR !!!!: {e}")
            return None

        # Lokal DB'ye ekle
        self.known_persons_db[new_pid] = {
            "master_vector": master,
            "count": len(vectors),
            "created": current_dt
        }

        print(f"CACHE ADD: P{new_pid} added to cache.")
        self.next_permanent_id += 1

        # Pending enrollments'a ekle
        self.pending_enrollments.append(new_pid)

        # Batch size'a ulaştıysak VEYA yeterli zaman geçtiyse rebuild tetikle
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

            # Arka planda rebuild (non-blocking)
            rebuild_thread = threading.Thread(target=self.rebuild_faiss_index, daemon=True)
            rebuild_thread.start()

        return new_pid

    def cleanup_stale_buffers(self, current_time):
        """
        Cleans up stale buffers for persons that haven't been seen recently.
        Prevents memory leaks in long-running deployments.
        """
        stale_pids = []

        # 1. recent_vector_buffers cleanup
        for pid in list(self.recent_vector_buffers.keys()):
            # Check if PID still exists in DB
            if pid not in self.known_persons_db:
                stale_pids.append(pid)
            # Or check last seen time
            elif (current_time - self.last_update_time_for_pid.get(pid, 0)) > self.BUFFER_MAX_IDLE_TIME:
                stale_pids.append(pid)

        # 2. last_update_time_for_pid cleanup
        for pid in list(self.last_update_time_for_pid.keys()):
            if pid not in self.known_persons_db:
                if pid not in stale_pids:  # Avoid duplicates
                    stale_pids.append(pid)

        # 3. Remove all stale entries
        if stale_pids:
            for pid in stale_pids:
                if pid in self.recent_vector_buffers:
                    del self.recent_vector_buffers[pid]
                if pid in self.last_update_time_for_pid:
                    del self.last_update_time_for_pid[pid]
                # last_known_box_for_pid de temizlenebilir (opsiyonel)
                if pid in self.last_known_box_for_pid:
                    del self.last_known_box_for_pid[pid]

            print(f"BUFFER CLEANUP: Removed {len(stale_pids)} stale buffer(s). "
                  f"Active buffers: {len(self.recent_vector_buffers)}")

    def incremental_update(self, perm_id, new_vector):
        """
        Updates master vector in DB cache using EMA (Exponential Moving Average).
        Index rebuild is handled separately in batches.
        """
        entry = self.known_persons_db.get(perm_id)
        if entry is None:
            return

        n = entry.get("count", 1)
        mongo_update_doc = {"$set": {"last_seen": datetime.now()}}

        # Recent vector buffer'ı kullan (ancak index'e eklemeden)
        buffer = self.recent_vector_buffers[perm_id]
        buffer.append(new_vector)

        # Vector update (MAX_VECTORS_FOR_UPDATE limitine kadar)
        if n < MAX_VECTORS_FOR_UPDATE:
            master = entry["master_vector"]

            # EMA with confidence weighting (daha stabil updates)
            alpha = 0.1  # Learning rate
            updated_master = self.l2_normalize((1 - alpha) * master + alpha * new_vector)

            new_count = n + 1
            mongo_update_doc["$set"]["master_vector"] = Binary(updated_master.tobytes())
            mongo_update_doc["$set"]["count"] = new_count

            entry["master_vector"] = updated_master
            entry["count"] = new_count

            if new_count == MAX_VECTORS_FOR_UPDATE:
                print(f"INFO: Person P{perm_id} vector locked after {new_count} updates.")

        try:
            self.persons_collection.update_one({"permanent_id": int(perm_id)}, mongo_update_doc)
        except Exception as e:
            print(f"!!!! MongoDB UPDATE ERROR (P{perm_id}) !!!!: {e}")

    def find_best_match(self, query_vector):
        """
        Performs a fast, thread-safe similarity search using the HNSW Faiss index.
        Now uses only SINGLE vector per person.
        """
        with self.faiss_lock:
            if self.faiss_index is None or self.faiss_index.ntotal == 0 or not self.index_to_pid_map:
                return None, -1.0

            try:
                query_vector_2d = query_vector.reshape(1, -1).astype('float32')
                D, I = self.faiss_index.search(query_vector_2d, k=1)
                best_similarity = D[0][0]

                if best_similarity >= self.current_match_threshold:
                    matched_index_position = I[0][0]
                    matched_pid = self.index_to_pid_map[matched_index_position]
                    return matched_pid, best_similarity
            except Exception as e:
                print(f"!!!! FAISS (HNSW) SEARCH ERROR !!!!: {e}")

        return None, -1.0

    def extract_embeddings_from_crops(self, crops):
        """Extracts feature embeddings (embeddings) from a list of cropped images."""
        rm = getattr(self.tracker, "reid_model", None)
        if rm is not None and hasattr(rm, 'model') and callable(rm.model):
            resized_crops = [cv2.resize(c, REID_INPUT_SIZE, interpolation=cv2.INTER_LINEAR) for c in crops if
                             c.size > 0]
            if not resized_crops: return []
            tensor_batch = torch.from_numpy(np.stack(resized_crops, axis=0)).permute(0, 3, 1, 2).float().to(
                self.device) / 255.0
            out = rm.model(tensor_batch)
            return [self.l2_normalize(v.cpu().numpy()) for v in out]

        print("!!!! WARNING: Re-ID model is not callable, returning random vectors! !!!!")
        return [self.l2_normalize(np.random.randn(VECTOR_DIMENSION).astype(np.float32)) for _ in crops]

    def detector_worker(self):
        """Runs the fast YOLOv8 detector in a separate thread."""
        print("Detector thread (YOLOv8) started...")
        while True:
            frame_to_process = self.frame_queue.get()
            if frame_to_process is None:
                print("Detector thread stopping...")
                break

            # YOLO, transform/tensor çevrimlerini kendisi halleder.
            # 'classes=0' sadece 'person' (insan) sınıfını arar.
            # 'conf' filtresini burada değil, aşağıda manuel yapacağız.
            output = self.detector(frame_to_process, classes=0, verbose=False)

            # YOLO'nun 'results' nesnesini 'boxmot'un beklediği numpy formatına çevir
            if not output:
                new_detections = np.empty((0, 6), dtype=np.float32)
            else:
                # Sadece ilk sonucu al (tek frame işliyoruz)
                boxes = output[0].boxes

                # 'boxes.data' içinde [x1, y1, x2, y2, score, class_id] bulunur
                # Faster R-CNN'in [boxes, scores, labels] formatından farklı

                # Canlı ayar (live-tuning) eşik değerine göre filtrele
                scores = boxes.conf.cpu().numpy()
                keep_mask = (scores >= self.current_detection_threshold)

                if keep_mask.any():
                    # 'boxmot' formatı: [x1, y1, x2, y2, score, class_id]
                    # 'class_id' zaten 0 (person) olduğu için direkt kullanabiliriz.
                    new_detections = boxes.data[keep_mask].cpu().numpy()

                    # Güvenlik kontrolü: Sütun sayısını 6 ile sınırla (eğer varsa tracker_id'leri kaldır)
                    if new_detections.shape[1] > 6:
                        new_detections = new_detections[:, :6]
                else:
                    new_detections = np.empty((0, 6), dtype=np.float32)

            try:
                self.detection_queue.get_nowait()
            except queue.Empty:
                pass
            self.detection_queue.put(new_detections)

    def run(self):
        """
        Runs the main application loop. Manages video stream, tracking,
        Re-ID, and user input.
        """
        detector_thread = threading.Thread(target=self.detector_worker, daemon=True)
        detector_thread.start()

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
                fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
                prev_time = current_time

                # --- 1. Detection (Asynchronous) ---
                if frame_count % DETECTION_INTERVAL == 0:
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass

                try:
                    detections = self.detection_queue.get_nowait()
                except queue.Empty:
                    pass

                # --- 2. Tracking (Fast) ---
                res = self.tracker.update(detections, frame)

                # --- 3. Batch Re-Identification (NEW EFFICIENT LOGIC) ---
                crops_to_process = []
                track_ids_for_crops = []
                tracker_results_list = []

                for *box, track_id, conf, cls, ind in res:
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    h, w, _ = frame.shape
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
                    all_vectors = self.extract_embeddings_from_crops(crops_to_process)

                vector_map = {}
                if len(all_vectors) == len(track_ids_for_crops):
                    vector_map = {tid: vec for tid, vec in zip(track_ids_for_crops, all_vectors)}
                else:
                    print(
                        f"!!!! WARNING: Mismatch in batch Re-ID. Crops: {len(crops_to_process)}, Vectors: {len(all_vectors)}")

                # --- 4. Process Results (using pre-computed vectors) ---
                current_frame_track_ids = set()
                for (box, track_id, conf, cls, ind, crop) in tracker_results_list:
                    current_frame_track_ids.add(track_id)
                    current_vector = vector_map.get(track_id)
                    if current_vector is None:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    label, color = f"T{track_id}", (0, 165, 255)

                    if track_id in self.track_id_to_permanent_id_map:
                        pid = self.track_id_to_permanent_id_map[track_id]
                        self.last_known_box_for_pid[pid] = box
                        label, color = f"P{pid}", (0, 255, 0)
                        if (time.time() - self.last_update_time_for_pid[pid]) > self.UPDATE_INTERVAL:
                            self.incremental_update(pid, current_vector)
                            self.last_update_time_for_pid[pid] = time.time()
                    else:
                        new_box_center = self.get_box_center(box)
                        spatial_match_pid = None
                        min_dist = float('inf')

                        # Kayıp buffer'ındaki her 'pid'yi kontrol et
                        for pid_to_check, data in self.lost_pid_buffer.items():
                            lost_box_center = self.get_box_center(data["last_box"])
                            dist = np.linalg.norm(np.array(new_box_center) - np.array(lost_box_center))

                            if dist < SPATIAL_MATCH_THRESHOLD and dist < min_dist:
                                min_dist = dist
                                spatial_match_pid = pid_to_check

                        if spatial_match_pid is not None:
                            # MEKANSAL EŞLEŞME BULUNDU! Faiss'e sorma.
                            self.track_id_to_permanent_id_map[track_id] = spatial_match_pid
                            label, color = f"P{spatial_match_pid} (Re-Acquired)", (0, 255, 128)  # Yeniden yakalandı rengi

                            # Buffer'dan çıkar (artık kayıp değil) ve son konumu güncelle
                            del self.lost_pid_buffer[spatial_match_pid]
                            self.last_known_box_for_pid[spatial_match_pid] = box
                            print(f"DEBUG: P{spatial_match_pid} spatially re-acquired by T{track_id}.")
                        else:
                            match_id, match_sim = self.find_best_match(current_vector)

                            if match_id is not None:
                                self.track_id_to_permanent_id_map[track_id] = match_id
                                label, color = f"P{match_id} (S:{match_sim:.2f})", (0, 255, 0)
                            else:
                                # --- ADVANCED Enrollment Quality Check ---
                                is_good_quality, reason, quality_score = self.advanced_quality_check(
                                    crop, box, frame.shape
                                )

                                if len(self.known_persons_db) > 0:
                                    print(
                                        f"DEBUG: New person (T{track_id}). Best sim: {match_sim:.4f}. "
                                        f"Quality: {quality_score:.2f} | Reason: {reason}")

                                buf = self.active_track_buffers[track_id]
                                if is_good_quality:
                                    buf["vectors"].append(current_vector)
                                    buf["quality_scores"] = buf.get("quality_scores", [])
                                    buf["quality_scores"].append(quality_score)

                                    # Show average quality in label
                                    avg_quality = np.mean(buf["quality_scores"])
                                    label = f"New? T{track_id} ({len(buf['vectors'])}/{MIN_ENROLL_FRAMES}) Q:{avg_quality:.2f}"
                                else:
                                    label = f"New? T{track_id} (REJECT: {reason})"

                                buf["last_seen"] = time.time()

                                # ENROLLMENT with quality threshold
                                if len(buf["vectors"]) >= MIN_ENROLL_FRAMES:
                                    avg_quality = np.mean(buf.get("quality_scores", [0.5]))

                                    # Require average quality ≥ 0.65 for enrollment
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

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- 5. Cleanup ---
                now = time.time()

                # 1. Aktif 'T...' buffer'larını temizle (bu mantık değişmedi)
                stale_active_tracks = [tid for tid, b in self.active_track_buffers.items() if
                                       (now - b.get("last_seen", now)) > 5.0]
                for tid in stale_active_tracks:
                    del self.active_track_buffers[tid]

                disappeared_track_ids = [tid for tid in self.track_id_to_permanent_id_map if
                                         tid not in current_frame_track_ids]

                for tid in disappeared_track_ids:
                    pid = self.track_id_to_permanent_id_map.pop(tid, None)  # Eşleşmeyi kaldır
                    if pid:
                        # 'recent_vector_buffers'dan SİLME!

                        # 'lost' buffer'ına ekle
                        if pid in self.last_known_box_for_pid:
                            self.lost_pid_buffer[pid] = {
                                "disappear_time": now,
                                "last_box": self.last_known_box_for_pid[pid]
                            }
                            # print(f"DEBUG: P{pid} lost, added to short-term memory.") # İsteğe bağlı debug
                        else:
                            # Bu pid için hiç kutu kaydedilmemişse (nadir durum), direkt sil
                            if pid in self.recent_vector_buffers: del self.recent_vector_buffers[pid]

                # Hafıza süresi (TTL) dolan 'pid'leri bul
                stale_lost_pids = [pid for pid, data in self.lost_pid_buffer.items() if
                                   (now - data["disappear_time"]) > SHORT_TERM_MEMORY_TTL]

                for pid in stale_lost_pids:
                    # print(f"DEBUG: P{pid} expired from short-term memory. Fully removing.") # İsteğe bağlı debug
                    del self.lost_pid_buffer[pid]

                    # Artık bu 'pid'yi diğer yerlerden de güvenle silebiliriz
                    if pid in self.recent_vector_buffers:
                        del self.recent_vector_buffers[pid]
                    if pid in self.last_known_box_for_pid:
                        del self.last_known_box_for_pid[pid]

                if (now - self.last_buffer_cleanup) > self.BUFFER_CLEANUP_INTERVAL:
                    self.cleanup_stale_buffers(now)
                    self.last_buffer_cleanup = now

                hud_text1 = f"FPS: {fps:.1f} | Detection (U/J): {self.current_detection_threshold:.2f}"
                hud_text2 = f"Match (Y/H): {self.current_match_threshold:.2f} | DB Size: {len(self.known_persons_db)}"

                # Draw a black background rectangle for readability
                cv2.rectangle(frame, (5, 5), (450, 60), (0, 0, 0), -1)

                # Draw the text
                cv2.putText(frame, hud_text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, hud_text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("ReID Vectors (MongoDB - Faiss)", frame)

                key = self.handle_keyboard_input()
                if key == 'q':
                    break

    def handle_keyboard_input(self):
        """Manages keyboard shortcuts, now including live tuning."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            self.frame_queue.put(None)
            return 'q'

        # --- NEW: Live Tuning Keyboard Controls ---

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

        # --- END NEW ---

        elif key == ord('r'):
            print("RESET: Deleting all entries from MongoDB...")
            try:
                self.persons_collection.delete_many({})
                self.faiss_collection.delete_many({})
                print("MongoDB collections have been cleared.")
            except Exception as e:
                print(f"!!!! MongoDB RESET ERROR !!!!: {e}")
            self.known_persons_db.clear()
            self.next_permanent_id = 1
            self.active_track_buffers.clear()
            self.track_id_to_permanent_id_map.clear()
            self.lost_pid_buffer.clear()
            self.last_known_box_for_pid.clear()
            self.recent_vector_buffers.clear()
            self.last_update_time_for_pid.clear()
            self.rebuild_faiss_index()
            print("Program memory (cache and index) has been cleared.")

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

        elif key == ord('k'):
            if self.known_persons_db:
                try:
                    last_id = max(self.known_persons_db.keys())
                    self.persons_collection.delete_one({"permanent_id": int(last_id)})
                    print(f"DELETED (k): P{last_id} deleted from MongoDB.")
                    del self.known_persons_db[last_id]

                    if last_id in self.recent_vector_buffers:
                        del self.recent_vector_buffers[last_id]
                    if last_id in self.last_update_time_for_pid:
                        del self.last_update_time_for_pid[last_id]
                    if last_id in self.last_known_box_for_pid:
                        del self.last_known_box_for_pid[last_id]

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
        """Releases all resources when the application closes."""
        self.stop_event.set()
        print("Cleaning up and exiting...")
        cv2.destroyAllWindows()
        self.mongo_client.close()
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