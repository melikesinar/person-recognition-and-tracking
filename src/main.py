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
from boxmot import BoostTrack
from datetime import datetime
from collections import deque
from bson.binary import Binary
from dotenv import load_dotenv
from collections import defaultdict
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights as Weights
)

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
        self.next_permanent_id = 1

        # These will be modified by keyboard presses
        self.current_match_threshold = MATCH_SIMILARITY_THRESHOLD
        self.current_detection_threshold = DETECTION_SCORE_THRESHOLD

        # --- Thread Communication Queues ---
        self.frame_queue = queue.Queue(maxsize=1)
        self.detection_queue = queue.Queue(maxsize=1)

        # --- Tracking and Enrollment Buffers ---
        self.active_track_buffers = defaultdict(lambda: {"vectors": [], "crops": [], "last_seen": time.time()})
        self.track_id_to_permanent_id_map = {}
        self.recent_vector_buffers = defaultdict(lambda: deque(maxlen=RECENT_VECTOR_BUFFER_SIZE))
        self.last_update_time_for_pid = defaultdict(float)
        self.UPDATE_INTERVAL = 1.0  # 1 second

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

        print("Loading Object Detector...")
        weights = Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
        self.detector.to(self.device).eval()
        self.transform = weights.transforms()

        print("Loading Re-ID Model...")
        self.tracker = BoostTrack(reid_weights=Path('osnet_x1_0_msmt17.pt'),
                                  device=(0 if self.device.type == 'cuda' else 'cpu'),
                                  half=False, with_reid=True)
        if not hasattr(self.tracker, 'reid_model') or self.tracker.reid_model is None:
            raise RuntimeError(f"Re-ID model failed to load! Exiting.")
        else:
            print("SUCCESS: Re-ID MODEL LOADED.")

        self.stop_event = threading.Event()
        self.rebuild_interval_seconds = 10  # 5 dakikada bir yeniden oluştur
        self.rebuilder_thread = threading.Thread(target=self.periodic_rebuild_worker, daemon=True)

        # --- Load Data ---
        self.load_db()

        # Thread'i DB yüklendikten SONRA başlat
        self.rebuilder_thread.start()

    def rebuild_faiss_index(self):
        """
        Rebuilds the in-memory Faiss index from the local DB cache in a thread-safe manner
        using a fast HNSW index.
        """
        with self.faiss_lock:
            print("Rebuilding HNSW Faiss index...")

            all_vectors = []

            temp_pid_map = []

            sorted_pids = sorted(self.known_persons_db.keys())
            for pid in sorted_pids:
                data = self.known_persons_db[pid]

                all_vectors.append(data['master_vector'])
                temp_pid_map.append(pid)  # 0. vektör -> bu pid

                all_vectors.append(data.get('recent_master_vector', data['master_vector']))
                temp_pid_map.append(pid)  # 1. vektör -> bu pid

            # HNSW indeksi oluştur (M=32, HNSW için standart bir parametredir)
            base_index = faiss.IndexHNSWFlat(VECTOR_DIMENSION, 32, faiss.METRIC_INNER_PRODUCT)
            # Performans için ayar:
            base_index.hnsw.efConstruction = 40

            if not all_vectors:
                print("No vectors to index. HNSW index initialized empty.")
                self.faiss_index = base_index  # Boş ama geçerli bir HNSW indeksi
                self.index_to_pid_map = []  # Harita da boş

                try:
                    self.faiss_collection.delete_one({"name": FAISS_INDEX_KEY})
                except Exception as e:
                    print(f"!!!! FAISS (empty) index delete error !!!!: {e}")
                return  # Bitti

            vector_matrix = np.stack(all_vectors, axis=0).astype('float32')

            # HNSW indeksini "eğit" (veriyi nasıl organize edeceğini öğrenir)
            base_index.train(vector_matrix)
            # Veriyi ekle
            base_index.add(vector_matrix)

            # Ana nesneleri ata
            self.faiss_index = base_index
            self.index_to_pid_map = temp_pid_map  # Haritayı güncelle

            print(f"HNSW Faiss index rebuilt successfully with {self.faiss_index.ntotal} vectors.")

            try:
                faiss.write_index(self.faiss_index, TEMP_FAISS_FILENAME)
                with open(TEMP_FAISS_FILENAME, "rb") as f:
                    index_buffer = f.read()
                self.faiss_collection.update_one(
                    {"name": FAISS_INDEX_KEY},
                    {"$set": {"index_data": Binary(index_buffer), "last_updated": datetime.now()}},
                    upsert=True
                )
                print(f"HNSW Faiss index saved to MongoDB ('{MONGO_FAISS_COLLECTION_NAME}').")
            except Exception as e:
                print(f"!!!! FAISS SAVE ERROR !!!!: {e}")
            finally:
                if os.path.exists(TEMP_FAISS_FILENAME):
                    os.remove(TEMP_FAISS_FILENAME)

    def periodic_rebuild_worker(self):
        """
        A background thread worker that periodically rebuilds the Faiss HNSW index
        to ensure it stays in sync with the database cache.
        """
        print(f"Background Faiss rebuilder started. Interval: {self.rebuild_interval_seconds} seconds.")
        while not self.stop_event.wait(self.rebuild_interval_seconds):
            print("\n--- PERIODIC REBUILD TRIGGERED ---")
            try:
                # 'rebuild_faiss_index' zaten thread-safe (faiss_lock kullanıyor)
                self.rebuild_faiss_index()
            except Exception as e:
                print(f"!!!! PERIODIC REBUILD FAILED !!!!: {e}")
            print("--- PERIODIC REBUILD FINISHED ---\n")

        print("Background Faiss rebuilder stopping.")

    def calculate_sharpness(self, image):
        """Calculates the sharpness of an image using Laplacian variance."""
        if image is None or image.size == 0:
            return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

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
                        "recent_master_vector": np.frombuffer(doc.get('recent_master_vector', doc['master_vector']),
                                                              dtype=np.float32),
                        "count": doc['count'],
                        "created": doc['created']
                    }
                self.next_permanent_id = max(self.known_persons_db.keys()) + 1 if self.known_persons_db else 1
                print(
                    f"Successfully loaded {len(self.known_persons_db)} persons from MongoDB. Next ID: {self.next_permanent_id}")
            else:
                print("No person records found in MongoDB. Starting fresh.")
                self.next_permanent_id = 1

            index_doc = self.faiss_collection.find_one({"name": FAISS_INDEX_KEY})
            if index_doc and 'index_data' in index_doc:
                print("Loading Faiss index from MongoDB...")
                try:
                    index_buffer = index_doc['index_data']
                    with open(TEMP_FAISS_FILENAME, "wb") as f:
                        f.write(index_buffer)
                    loaded_index = faiss.read_index(TEMP_FAISS_FILENAME)

                    # HNSW için efSearch ayarı (arama hızı/doğruluk dengesi)
                    if hasattr(loaded_index, 'hnsw'):
                        loaded_index.hnsw.efSearch = 64

                    if loaded_index.ntotal != len(self.known_persons_db) * 2:
                        print("Faiss index is stale (vector count mismatch). Rebuilding...")
                        raise ValueError("Index stale")
                    with self.faiss_lock:
                        self.faiss_index = loaded_index
                        # Manuel haritayı da yeniden oluştur
                        print("Rebuilding index-to-pid map from loaded DB...")
                        temp_pid_map = []
                        sorted_pids = sorted(self.known_persons_db.keys())
                        for pid in sorted_pids:
                            temp_pid_map.append(pid)  # master_vector
                            temp_pid_map.append(pid)  # recent_master_vector
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

    def l2_normalize(self, v):
        """Normalizes a vector to unit length (L2 norm)."""
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    def enroll_person(self, vectors):
        """
        Enrolls a new person into the MongoDB and local cache.
        The HNSW index is NOT updated here; it is updated by a periodic rebuilder.
        """
        if not vectors: raise ValueError("No vectors provided to enroll.")
        master = self.l2_normalize(np.mean(np.stack(vectors, axis=0), axis=0))
        current_dt = datetime.now()

        new_pid = self.next_permanent_id  # ID'yi önce alalım

        new_person_doc = {"permanent_id": new_pid, "master_vector": Binary(master.tobytes()),
                          "recent_master_vector": Binary(master.tobytes()), "count": len(vectors),
                          "created": current_dt, "last_seen": current_dt}
        try:
            self.persons_collection.insert_one(new_person_doc)
            print(f"MongoDB INSERT: New person (P{new_pid}) added.")
        except Exception as e:
            print(f"!!!! MongoDB INSERT ERROR !!!!: {e}")
            return None  # Hata varsa ID döndürme

        # Yeni kişiyi lokal DB'ye ekle
        self.known_persons_db[new_pid] = {"master_vector": master, "recent_master_vector": master,
                                          "count": len(vectors), "created": current_dt}

        print(f"CACHE ADD: P{new_pid} önbelleğe eklendi. (İndekse {self.rebuild_interval_seconds}sn içinde eklenecek)")

        self.next_permanent_id += 1
        return new_pid  # Başarılı kayıtta yeni ID'yi döndür

    def incremental_update(self, perm_id, new_vector):
        """
        Updates vectors in DB cache.
        The HNSW Faiss index is NOT updated here; it is updated by a periodic rebuilder.
        """
        entry = self.known_persons_db.get(perm_id)
        if entry is None: return
        n = entry.get("count", 1)
        mongo_update_doc = {"$set": {"last_seen": datetime.now()}}
        buffer = self.recent_vector_buffers[perm_id]
        buffer.append(new_vector)
        updated_recent_master = self.l2_normalize(np.mean(buffer, axis=0))
        mongo_update_doc["$set"]["recent_master_vector"] = Binary(updated_recent_master.tobytes())
        entry["recent_master_vector"] = updated_recent_master
        if n < MAX_VECTORS_FOR_UPDATE:
            master = entry["master_vector"]
            updated_master = self.l2_normalize((n * master + new_vector) / (n + 1))
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
        """Performs a fast, thread-safe similarity search using the HNSW Faiss index."""
        with self.faiss_lock:
            # self.index_to_pid_map'in de dolu olduğunu kontrol et
            if self.faiss_index is None or self.faiss_index.ntotal == 0 or not self.index_to_pid_map:
                return None, -1.0
            try:
                query_vector_2d = query_vector.reshape(1, -1).astype('float32')
                D, I = self.faiss_index.search(query_vector_2d, k=1)
                best_similarity = D[0][0]
                if best_similarity >= self.current_match_threshold:
                    # Bu sıra numarasını manuel haritamızdan PID'ye çeviriyoruz.
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
        """Runs the slow detector in a separate thread."""
        print("Detector thread started...")
        while True:
            frame_to_process = self.frame_queue.get()
            if frame_to_process is None:
                print("Detector thread stopping...")
                break
            with torch.inference_mode():
                tensor = torch.from_numpy(cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).to(
                    torch.uint8)
                output = self.detector([self.transform(tensor).to(self.device)])[0]
                scores, labels, boxes = output['scores'], output['labels'], output['boxes']

                # --- UPDATED: Use the live-tuning threshold ---
                # Read the current threshold value from the main class
                keep_mask = (scores >= self.current_detection_threshold) & (labels == 1)

                new_detections = np.concatenate(
                    [boxes[keep_mask].cpu().numpy(), scores[keep_mask].cpu().numpy()[:, None],
                     labels[keep_mask].cpu().numpy()[:, None]], axis=1)
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
                        label, color = f"P{pid}", (0, 255, 0)
                        if (time.time() - self.last_update_time_for_pid[pid]) > self.UPDATE_INTERVAL:
                            self.incremental_update(pid, current_vector)
                            self.last_update_time_for_pid[pid] = time.time()
                    else:
                        match_id, match_sim = self.find_best_match(current_vector)

                        if match_id is not None:
                            self.track_id_to_permanent_id_map[track_id] = match_id
                            label, color = f"P{match_id} (S:{match_sim:.2f})", (0, 255, 0)
                        else:
                            # --- Enrollment Quality Check ---
                            bbox_area = crop.shape[0] * crop.shape[1]
                            sharpness = self.calculate_sharpness(crop)
                            is_good_quality = (bbox_area >= MIN_BBOX_AREA_ENROLL) and (
                                    sharpness >= MIN_SHARPNESS_ENROLL)

                            if len(self.known_persons_db) > 0:
                                print(
                                    f"DEBUG: New person (T{track_id}). Best sim: {match_sim:.4f}. Quality: {sharpness:.1f} / Area: {bbox_area}. Good: {is_good_quality}")

                            buf = self.active_track_buffers[track_id]
                            if is_good_quality:
                                buf["vectors"].append(current_vector)
                                label = f"New? T{track_id} ({len(buf['vectors'])})"
                            else:
                                reason = "Too small" if bbox_area < MIN_BBOX_AREA_ENROLL else "Too blurry"
                                label = f"New? T{track_id} (REJECT: {reason})"

                            buf["last_seen"] = time.time()
                            if len(buf["vectors"]) >= MIN_ENROLL_FRAMES:
                                new_pid = self.enroll_person(buf["vectors"])
                                if new_pid:  # Check if enrollment was successful
                                    print(f"ENROLLED: New person P{new_pid}. Vectors: {len(buf['vectors'])}")
                                    self.track_id_to_permanent_id_map[track_id] = new_pid
                                    del self.active_track_buffers[track_id]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- 5. Cleanup ---
                now = time.time()
                stale = [tid for tid, b in self.active_track_buffers.items() if (now - b.get("last_seen", now)) > 5.0]
                for tid in stale: del self.active_track_buffers[tid]

                disappeared = [tid for tid in self.track_id_to_permanent_id_map if tid not in current_frame_track_ids]
                for tid in disappeared:
                    pid = self.track_id_to_permanent_id_map.pop(tid, None)
                    if pid and pid in self.recent_vector_buffers:
                        del self.recent_vector_buffers[pid]

                # --- NEW: Draw the HUD (Heads-Up Display) ---
                hud_text1 = f"FPS: {fps:.1f} | Detection (U/J): {self.current_detection_threshold:.2f}"
                hud_text2 = f"Match (Y/H): {self.current_match_threshold:.2f} | DB Size: {len(self.known_persons_db)}"

                # Draw a black background rectangle for readability
                cv2.rectangle(frame, (5, 5), (450, 60), (0, 0, 0), -1)

                # Draw the text
                cv2.putText(frame, hud_text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, hud_text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # --- END NEW ---

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
                    self.active_track_buffers.clear()
                    self.track_id_to_permanent_id_map.clear()
                    self.rebuild_faiss_index()
                    print(f"DELETED (k): P{last_id} deleted from local cache. Index rebuilt.")
                except Exception as e:
                    print(f"Error deleting person: {e}")
            else:
                print("DB is already empty. Nothing to delete.")
        return key

    def cleanup(self):
        """Releases all resources when the application closes."""
        self.stop_event.set()  # Arka plan thread'ine durmasını söyle

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