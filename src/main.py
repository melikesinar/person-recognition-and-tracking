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

load_dotenv()


def main():
    """Main function to run the person Re-ID and tracking system."""

    # Constant values
    MONGO_CONNECTION_STRING = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("DB_NAME")
    MONGO_COLLECTION_NAME = os.getenv("COLLECTION")

    MIN_ENROLL_FRAMES = 30
    MATCH_SIMILARITY_THRESHOLD = 0.85
    MAX_VECTORS_FOR_UPDATE = 300
    RECENT_VECTOR_BUFFER_SIZE = 50
    DETECTION_INTERVAL = 2  # Run detector every 2 frames
    REID_INPUT_SIZE = (128, 256)
    VECTOR_DIMENSION = 512  # osnet_x1_0_msmt17.pt modelinin vektör boyutu

    faiss_index = None
    # Faiss'in döndürdüğü 'index' (0, 1, 2...) ile 'permanent_id' (P1, P2...) arasındaki harita
    index_to_pid_map = []

    # Thread communication queues
    frame_queue = queue.Queue(maxsize=1)
    detection_queue = queue.Queue(maxsize=1)

    # Database Connection
    try:
        mongo_client = pymongo.MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=5000)
        mongo_client.server_info()
        db = mongo_client[MONGO_DB_NAME]
        persons_collection = db[MONGO_COLLECTION_NAME]
        persons_collection.create_index("permanent_id", unique=True)
        print(f"Successfully connected to MongoDB. DB: '{MONGO_DB_NAME}', Collection: '{MONGO_COLLECTION_NAME}'")
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(f"!!!! MongoDB CONNECTION ERROR !!!!: {e}")
        exit()

    def rebuild_faiss_index(db_cache):
        """Rebuilds the in-memory Faiss index from the local DB cache."""
        nonlocal faiss_index, index_to_pid_map  # Dış scope'taki değişkenleri değiştirmek için
        print("Rebuilding Faiss index...")

        index_to_pid_map.clear()
        all_vectors = []

        # Her kişi için 2 vektörü de (master ve recent) indekse ekle
        # Bu, her iki görünüme karşı da arama yapmamızı sağlar
        sorted_pids = sorted(db_cache.keys())  # Tutarlılık için ID'leri sırala
        for pid in sorted_pids:
            data = db_cache[pid]
            # Master vektörü ekle
            all_vectors.append(data['master_vector'])
            index_to_pid_map.append(pid)

            # Recent vektörü ekle
            recent_vec = data.get('recent_master_vector', data['master_vector'])
            all_vectors.append(recent_vec)
            index_to_pid_map.append(pid)

        if not all_vectors:
            print("No vectors to index. Faiss index is empty.")
            faiss_index = None
            return

        # Vektörleri Faiss'in beklediği formata getir
        vector_matrix = np.stack(all_vectors, axis=0).astype('float32')

        # L2 normalize edilmiş vektörler için Inner Product (Kosinüs Benzerliği) en hızlı indekstir
        faiss_index = faiss.IndexFlatIP(VECTOR_DIMENSION)
        faiss_index.add(vector_matrix)
        print(f"Faiss index rebuilt successfully with {faiss_index.ntotal} vectors ({len(db_cache)} persons).")


    # --- Helper functions ---
    def load_db():
        """Loads all persons from MongoDB into a local cache and builds the initial Faiss index."""
        print("Loading database from MongoDB...")
        db_cache = {}
        try:
            all_persons = list(persons_collection.find({}))
            if not all_persons:
                print("No records found in MongoDB. Starting fresh.")
                rebuild_faiss_index(db_cache)  # Boş bir indeks oluştur
                return {}, 1

            for doc in all_persons:
                pid = doc['permanent_id']
                vec = np.frombuffer(doc['master_vector'], dtype=np.float32)
                recent_vec_data = doc.get('recent_master_vector', doc['master_vector'])
                recent_vec = np.frombuffer(recent_vec_data, dtype=np.float32)

                db_cache[pid] = {
                    "master_vector": vec,
                    "recent_master_vector": recent_vec,
                    "count": doc['count'],
                    "created": doc['created']
                }

            next_id = max(db_cache.keys()) + 1
            print(f"Successfully loaded {len(db_cache)} persons from MongoDB. Next ID: {next_id}")

            rebuild_faiss_index(db_cache)

        except Exception as e:
            print(f"Error loading DB, creating new one: {e}")
            rebuild_faiss_index({})  # Hata durumunda boş indeks oluştur
            return {}, 1

        return db_cache, next_id

    def l2_normalize(v):
        """Normalizes a vector to unit length (L2 norm)."""
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    def cosine_similarity(a, b):
        """Calculates cosine similarity between two L2-normalized vectors."""
        return float(np.dot(a, b))

    def enroll_person(db, next_id, vectors):
        """Enrolls a new person and triggers a Faiss index rebuild."""
        if len(vectors) == 0:
            raise ValueError("No vectors provided to enroll.")
        vecs = np.stack(vectors, axis=0)
        master = np.mean(vecs, axis=0)
        master = l2_normalize(master)
        current_dt = datetime.now()
        new_person_doc = {
            "permanent_id": next_id,
            "master_vector": Binary(master.tobytes()),
            "recent_master_vector": Binary(master.tobytes()),
            "count": vecs.shape[0],
            "created": current_dt,
            "last_seen": current_dt,
        }
        try:
            persons_collection.insert_one(new_person_doc)
            print(f"MongoDB INSERT: New person (P{next_id}) added to MongoDB.")
        except Exception as e:
            print(f"!!!! MongoDB INSERT ERROR !!!!: {e}")
            return db, next_id
        db[next_id] = {
            "master_vector": master,
            "recent_master_vector": master,
            "count": vecs.shape[0],
            "created": current_dt
        }

        rebuild_faiss_index(db)

        return next_id, db, next_id + 1

    def incremental_update(db, perm_id, new_vector):
        """
        Updates vectors in the DB cache. The Faiss index is NOT rebuilt here for performance.
        It will be updated on the next new enrollment.
        """
        entry = db.get(perm_id)
        if entry is None:
            db[perm_id] = {"master_vector": l2_normalize(new_vector), "count": 1, "created": datetime.now()}
            return db
        n = entry.get("count", 1)
        current_time = datetime.now()
        mongo_update_doc = {"$set": {"last_seen": current_time}}
        buffer = recent_vector_buffers[perm_id]
        buffer.append(new_vector)
        updated_recent_master = l2_normalize(np.mean(buffer, axis=0))
        mongo_update_doc["$set"]["recent_master_vector"] = Binary(updated_recent_master.tobytes())
        entry["recent_master_vector"] = updated_recent_master
        if n < MAX_VECTORS_FOR_UPDATE:
            master = entry["master_vector"]
            updated_master = (n * master + new_vector) / (n + 1)
            updated_master = l2_normalize(updated_master)
            new_count = n + 1
            mongo_update_doc["$set"]["master_vector"] = Binary(updated_master.tobytes())
            mongo_update_doc["$set"]["count"] = new_count
            entry["master_vector"] = updated_master
            entry["count"] = new_count
            if new_count == MAX_VECTORS_FOR_UPDATE:
                print(f"INFO: Person P{perm_id} has reached {new_count} vectors. Master vector is now locked.")
        try:
            persons_collection.update_one(
                {"permanent_id": perm_id},
                mongo_update_doc
            )
        except Exception as e:
            print(f"!!!! MongoDB UPDATE ERROR (P{perm_id}) !!!!: {e}")
        return db

    def find_best_match(query_vector, db):
        """Performs a fast similarity search using the Faiss index."""
        nonlocal faiss_index, index_to_pid_map

        # Eğer indeks boşsa veya hiç oluşturulmamışsa, eşleşme olamaz
        if faiss_index is None or faiss_index.ntotal == 0:
            return None, -1.0

        try:
            # Faiss 'search' metodu 2D bir array bekler, bu yüzden vektörü yeniden şekillendiriyoruz
            query_vector_2d = query_vector.reshape(1, -1).astype('float32')

            # k=1: Sadece en iyi 1 sonucu bul
            # D: Mesafeler (bizim durumumuzda Inner Product/Benzerlik)
            # I: En yakın vektörlerin indeksleri
            D, I = faiss_index.search(query_vector_2d, k=1)

            best_match_index = I[0][0]
            best_similarity = D[0][0]

            if best_similarity >= MATCH_SIMILARITY_THRESHOLD:
                # Faiss indeksinden (0, 1, 2...) permanent_id'ye (P1, P2...) geçiş yap
                matched_pid = index_to_pid_map[best_match_index]
                return matched_pid, best_similarity

        except Exception as e:
            print(f"!!!! FAISS SEARCH ERROR !!!!: {e}")

        return None, -1.0  # Hata durumunda veya eşik altında kalırsa

    # --- Main Execution ---

    # Load DB
    known_persons_db, next_permanent_id = load_db()
    print(f"Local cache loaded: {len(known_persons_db)} persons. next_id={next_permanent_id}")

    # Set device and models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    weights = Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    detector.to(device).eval()
    transform = weights.transforms()

    # Initialize the tracker
    tracker = BoostTrack(reid_weights=Path('osnet_x1_0_msmt17.pt'), device=(0 if device.type == 'cuda' else device),
                         half=False, with_reid=True)

    if not hasattr(tracker, 'reid_model') or tracker.reid_model is None:
        raise RuntimeError(f"Re-ID model failed to load! Exiting.")
    else:
        print("SUCCESS: Re-ID MODEL LOADED SUCCESSFULLY.")

    # Buffers & maps
    active_track_buffers = defaultdict(lambda: {"vectors": [], "crops": [], "last_seen": time.time()})
    track_id_to_permanent_id_map = {}
    recent_vector_buffers = defaultdict(lambda: deque(maxlen=RECENT_VECTOR_BUFFER_SIZE))
    last_update_time_for_pid = defaultdict(float)
    UPDATE_INTERVAL = 1.0

    prev_time = 0
    frame_count = 0

    def extract_embeddings_from_crops(crops):
        """Extracts feature embeddings from a list of cropped images."""
        rm = getattr(tracker, "reid_model", None)
        if rm is not None and hasattr(rm, 'model') and callable(rm.model):
            resized_crops = []
            for crop in crops:
                if crop.size == 0: continue
                resized = cv2.resize(crop, REID_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
                resized_crops.append(resized)
            if not resized_crops:
                return []
            np_batch = np.stack(resized_crops, axis=0)
            tensor_batch = torch.from_numpy(np_batch).permute(0, 3, 1, 2).float().to(device)
            tensor_batch /= 255.0
            out = rm.model(tensor_batch)
            vecs = []
            for v in out:
                if torch.is_tensor(v):
                    v = v.cpu().numpy()
                v = l2_normalize(np.asarray(v, dtype=np.float32))
                vecs.append(v)
            return vecs
        print("!!!! WARNING: Re-ID model is not callable, returning random vectors! !!!!")
        vec = l2_normalize(np.random.randn(VECTOR_DIMENSION).astype(np.float32))
        return [vec for _ in crops]

    # --- Detector Thread ---
    def detector_worker():
        """Runs the slow detector in a separate thread."""
        print("Detector thread started...")
        while True:
            frame_to_process = frame_queue.get()
            if frame_to_process is None:
                print("Detector thread stopping...")
                break
            with torch.inference_mode():
                rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.uint8)
                input_tensor = transform(tensor).to(device)
                output = detector([input_tensor])[0]
                scores = output['scores']
                labels = output['labels']
                boxes = output['boxes']
                person_label = 1
                score_threshold = 0.7
                keep_mask = (scores >= score_threshold) & (labels == person_label)
                boxes_filtered = boxes[keep_mask].cpu().numpy()
                scores_filtered = scores[keep_mask].cpu().numpy()
                labels_filtered = labels[keep_mask].cpu().numpy()
                new_detections = np.concatenate(
                    [boxes_filtered, scores_filtered[:, None], labels_filtered[:, None]], axis=1)
                try:
                    detection_queue.get_nowait()
                except queue.Empty:
                    pass
                detection_queue.put(new_detections)

    # Start the detector thread
    detector_thread = threading.Thread(target=detector_worker, daemon=True)
    detector_thread.start()

    # Video loop
    cap = cv2.VideoCapture(0)
    with torch.inference_mode():
        detections = np.empty((0, 6), dtype=np.float32)
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            current_time = time.time()
            if prev_time > 0:
                fps = 1 / (current_time - prev_time)
            else:
                fps = 0
            prev_time = current_time
            h, w = frame.shape[:2]
            if frame_count % DETECTION_INTERVAL == 0:
                try:
                    frame_copy = frame.copy()
                    frame_queue.put_nowait(frame_copy)
                except queue.Full:
                    pass
            try:
                detections = detection_queue.get_nowait()
            except queue.Empty:
                pass
            res = tracker.update(detections, frame)

            if res.shape[0] == 0:
                pass
            current_frame_track_ids = set()
            for *box, track_id, conf, cls, ind in res:
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                current_frame_track_ids.add(track_id)
                x1 = max(0, min(w - 1, x1));
                x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1));
                y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1: continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                label_to_display = f"T{track_id}"
                box_color = (0, 165, 255)
                if track_id in track_id_to_permanent_id_map:
                    pid = track_id_to_permanent_id_map[track_id]
                    label_to_display = f"P{pid}"
                    box_color = (0, 255, 0)
                    current_time_for_update = time.time()
                    if (current_time_for_update - last_update_time_for_pid[pid]) > UPDATE_INTERVAL:
                        vecs = extract_embeddings_from_crops([crop])
                        if vecs:
                            current_vector = vecs[0]
                            known_persons_db = incremental_update(known_persons_db, pid, current_vector)
                            last_update_time_for_pid[pid] = current_time_for_update
                else:
                    vecs = extract_embeddings_from_crops([crop])
                    if not vecs: continue
                    current_vector = vecs[0]
                    match_id, match_sim = find_best_match(current_vector, known_persons_db)  # Artık Faiss kullanıyor
                    if match_id is not None:
                        track_id_to_permanent_id_map[track_id] = match_id
                        label_to_display = f"P{match_id} (S:{match_sim:.2f})"
                        box_color = (0, 255, 0)
                    else:
                        if len(known_persons_db) > 0:
                            print(f"DEBUG: New person (T{track_id}). Best similarity: {match_sim:.4f}")
                        buf = active_track_buffers[track_id]
                        buf["vectors"].append(current_vector)
                        buf["crops"].append(crop)
                        buf["last_seen"] = time.time()
                        label_to_display = f"New? T{track_id} ({len(buf['vectors'])})"
                        if len(buf["vectors"]) >= MIN_ENROLL_FRAMES:
                            new_pid, known_persons_db, next_permanent_id = enroll_person(known_persons_db,
                                                                                         next_permanent_id,
                                                                                         buf["vectors"])
                            print("Enrolled new person:", new_pid, "vectors:", len(buf["vectors"]))
                            track_id_to_permanent_id_map[track_id] = new_pid
                            del active_track_buffers[track_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label_to_display, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # --- cleanup ---
            now = time.time()
            stale = [tid for tid, b in active_track_buffers.items() if now - b.get("last_seen", now) > 5.0]
            for tid in stale:
                del active_track_buffers[tid]

            disappeared = [tid for tid in list(track_id_to_permanent_id_map.keys()) if
                           tid not in current_frame_track_ids]
            for tid in disappeared:
                pid = track_id_to_permanent_id_map.pop(tid, None)
                if pid and pid in recent_vector_buffers:
                    del recent_vector_buffers[pid]

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("ReID Vectors (MongoDB - Faiss)", frame)

            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                frame_queue.put(None)
                break

            elif key == ord('r'):
                print("RESET: Deleting all entries from MongoDB...")
                try:
                    persons_collection.delete_many({})
                    print("MongoDB 'persons' collection has been cleared.")
                except Exception as e:
                    print(f"!!!! MongoDB RESET ERROR !!!!: {e}")
                known_persons_db.clear()
                next_permanent_id = 1
                active_track_buffers.clear()
                track_id_to_permanent_id_map.clear()
                rebuild_faiss_index(known_persons_db)
                print("Program memory (cache) has been cleared.")

            elif key == ord('d'):
                print("\n--- DB STATUS (d) (Local Cache) ---")
                print(f"Persons in cache: {len(known_persons_db)}")
                print(f"Vectors in Faiss index: {faiss_index.ntotal if faiss_index else 0}")
                print(f"Next available ID: {next_permanent_id}")
                for pid, data in known_persons_db.items():
                    created_date = data.get('created', 'N/A')
                    if isinstance(created_date, datetime):
                        created_date = created_date.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"  - P{pid}: {data['count']} vectors, Created: {created_date}")
                print("------------------ Check MongoDB Compass for the full list of records.\n")

            elif key == ord('k'):
                if known_persons_db:
                    try:
                        last_id = max(known_persons_db.keys())
                        persons_collection.delete_one({"permanent_id": last_id})
                        print(f"DELETED (k): P{last_id} deleted from MongoDB.")
                        del known_persons_db[last_id]
                        active_track_buffers.clear()
                        track_id_to_permanent_id_map.clear()
                        rebuild_faiss_index(known_persons_db)
                        print(f"DELETED (k): P{last_id} deleted from local cache. Maps cleared.")
                    except Exception as e:
                        print(f"Error deleting person: {e}")
                else:
                    print("DB is already empty. Nothing to delete.")

    # final cleanup
    cap.release()
    cv2.destroyAllWindows()
    mongo_client.close()
    print("Exited cleanly. DB entries (in cache):", len(known_persons_db))


if __name__ == "__main__":
    main()