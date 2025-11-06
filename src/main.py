# main_reid_vectors.py
import os
import cv2
import time
import queue
import torch
import pymongo
import threading
import numpy as np
import faiss
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

    # --- Constants ---
    MONGO_CONNECTION_STRING = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("DB_NAME")
    MONGO_COLLECTION_NAME = os.getenv("COLLECTION")
    MONGO_FAISS_COLLECTION_NAME = "faiss_indexes"  # New collection to store the index

    MIN_ENROLL_FRAMES = 30
    MATCH_SIMILARITY_THRESHOLD = 0.85
    MAX_VECTORS_FOR_UPDATE = 300
    RECENT_VECTOR_BUFFER_SIZE = 50
    DETECTION_INTERVAL = 2
    REID_INPUT_SIZE = (128, 256)
    VECTOR_DIMENSION = 512

    # --- Faiss and Threading Globals ---
    faiss_index = None
    faiss_lock = threading.Lock()  # Lock for thread-safe access to the Faiss index

    # --- Thread communication queues ---
    frame_queue = queue.Queue(maxsize=1)
    detection_queue = queue.Queue(maxsize=1)

    # --- Database Connection ---
    try:
        mongo_client = pymongo.MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=5000)
        mongo_client.server_info()
        db = mongo_client[MONGO_DB_NAME]
        persons_collection = db[MONGO_COLLECTION_NAME]
        faiss_collection = db[MONGO_FAISS_COLLECTION_NAME]  # New handle for the index collection
        persons_collection.create_index("permanent_id", unique=True)
        print(
            f"Successfully connected to MongoDB. DB: '{MONGO_DB_NAME}', Collections: '{MONGO_COLLECTION_NAME}', '{MONGO_FAISS_COLLECTION_NAME}'")
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(f"!!!! MongoDB CONNECTION ERROR !!!!: {e}")
        exit()

    # --- Helper Functions ---

    def rebuild_faiss_index(db_cache):
        """
        Rebuilds the in-memory Faiss index from the local DB cache in a thread-safe manner
        and saves the serialized index to MongoDB via a temporary file.
        """
        nonlocal faiss_index
        with faiss_lock:  # Ensure exclusive access to the index
            print("Rebuilding Faiss index...")

            all_vectors = []
            all_pids_for_vectors = []
            sorted_pids = sorted(db_cache.keys())
            for pid in sorted_pids:
                data = db_cache[pid]
                all_vectors.append(data['master_vector'])
                all_pids_for_vectors.append(pid)
                all_vectors.append(data.get('recent_master_vector', data['master_vector']))
                all_pids_for_vectors.append(pid)

            if not all_vectors:
                print("No vectors to index. Faiss index is empty.")
                faiss_index = None
                faiss_collection.delete_one({"name": "primary_person_index"})
                return

            vector_matrix = np.stack(all_vectors, axis=0).astype('float32')
            id_matrix = np.array(all_pids_for_vectors).astype('int64')

            base_index = faiss.IndexFlatIP(VECTOR_DIMENSION)
            faiss_index = faiss.IndexIDMap(base_index)
            faiss_index.add_with_ids(vector_matrix, id_matrix)
            print(f"Faiss index rebuilt successfully with {faiss_index.ntotal} vectors.")

            temp_index_filename = "temp_faiss.index"
            try:
                # 1. Write index to a temporary file
                faiss.write_index(faiss_index, temp_index_filename)

                # 2. Read the bytes from that file
                with open(temp_index_filename, "rb") as f:
                    index_buffer = f.read()

                # 3. Save those bytes to MongoDB
                faiss_collection.update_one(
                    {"name": "primary_person_index"},
                    {"$set": {"index_data": Binary(index_buffer), "last_updated": datetime.now()}},
                    upsert=True
                )
                print(f"Faiss index saved to MongoDB collection '{MONGO_FAISS_COLLECTION_NAME}'.")

            except Exception as e:
                print(f"!!!! FAISS SAVE ERROR !!!!: {e}")
            finally:
                # 4. Clean up the temporary file
                if os.path.exists(temp_index_filename):
                    os.remove(temp_index_filename)

    def load_db():
        """
        Loads all person vectors from MongoDB into a local cache and
        loads the Faiss index from MongoDB (or rebuilds it if necessary).
        """
        nonlocal faiss_index
        print("Loading database from MongoDB...")
        db_cache = {}
        next_id = 1
        try:
            # Load person vectors into cache
            all_persons = list(persons_collection.find({}))
            if all_persons:
                for doc in all_persons:
                    pid = doc['permanent_id']
                    db_cache[pid] = {
                        "master_vector": np.frombuffer(doc['master_vector'], dtype=np.float32),
                        "recent_master_vector": np.frombuffer(doc.get('recent_master_vector', doc['master_vector']),
                                                              dtype=np.float32),
                        "count": doc['count'],
                        "created": doc['created']
                    }
                next_id = max(db_cache.keys()) + 1 if db_cache else 1
                print(f"Successfully loaded {len(db_cache)} persons from MongoDB. Next ID: {next_id}")
            else:
                print("No person records found in MongoDB. Starting fresh.")

            # Attempt to load Faiss index from MongoDB
            index_doc = faiss_collection.find_one({"name": "primary_person_index"})
            if index_doc and 'index_data' in index_doc:
                print("Loading Faiss index from MongoDB...")
                try:
                    index_buffer = index_doc['index_data']
                    loaded_index = faiss.read_index_buf(index_buffer)
                    # Verification
                    if loaded_index.ntotal != len(db_cache) * 2:
                        print("Faiss index is stale (vector count mismatch). Rebuilding...")
                        raise ValueError("Index stale")
                    with faiss_lock:
                        faiss_index = loaded_index
                    print("Faiss index loaded successfully from MongoDB.")
                except Exception as e:
                    print(f"Failed to load Faiss index from MongoDB ({e}), rebuilding...")
                    rebuild_faiss_index(db_cache)
            else:
                print("No Faiss index found in MongoDB. Building from scratch...")
                rebuild_faiss_index(db_cache)

        except Exception as e:
            print(f"Error loading DB, creating new one: {e}")
            rebuild_faiss_index({})
            return {}, 1

        return db_cache, next_id

    def l2_normalize(v):
        """Normalizes a vector to unit length (L2 norm)."""
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    def enroll_person(db, next_id, vectors):
        """Enrolls a new person and triggers a Faiss index rebuild."""
        if not vectors: raise ValueError("No vectors provided to enroll.")
        master = l2_normalize(np.mean(np.stack(vectors, axis=0), axis=0))
        current_dt = datetime.now()
        new_person_doc = {"permanent_id": next_id, "master_vector": Binary(master.tobytes()),
                          "recent_master_vector": Binary(master.tobytes()), "count": len(vectors),
                          "created": current_dt, "last_seen": current_dt}
        try:
            persons_collection.insert_one(new_person_doc)
            print(f"MongoDB INSERT: New person (P{next_id}) added.")
        except Exception as e:
            print(f"!!!! MongoDB INSERT ERROR !!!!: {e}")
            return db, next_id
        db[next_id] = {"master_vector": master, "recent_master_vector": master, "count": len(vectors),
                       "created": current_dt}
        rebuild_faiss_index(db)
        return next_id, db, next_id + 1

    def incremental_update(db, perm_id, new_vector):
        """
        Updates vectors in the DB cache. The Faiss index is NOT rebuilt here for performance.
        """
        entry = db.get(perm_id)
        if entry is None: return db
        n = entry.get("count", 1)
        mongo_update_doc = {"$set": {"last_seen": datetime.now()}}
        buffer = recent_vector_buffers[perm_id]
        buffer.append(new_vector)
        updated_recent_master = l2_normalize(np.mean(buffer, axis=0))
        mongo_update_doc["$set"]["recent_master_vector"] = Binary(updated_recent_master.tobytes())
        entry["recent_master_vector"] = updated_recent_master
        if n < MAX_VECTORS_FOR_UPDATE:
            master = entry["master_vector"]
            updated_master = l2_normalize((n * master + new_vector) / (n + 1))
            new_count = n + 1
            mongo_update_doc["$set"]["master_vector"] = Binary(updated_master.tobytes())
            mongo_update_doc["$set"]["count"] = new_count
            entry["master_vector"] = updated_master
            entry["count"] = new_count
            if new_count == MAX_VECTORS_FOR_UPDATE:
                print(f"INFO: Person P{perm_id} vector locked after {new_count} updates.")
        try:
            persons_collection.update_one({"permanent_id": int(perm_id)}, mongo_update_doc)
        except Exception as e:
            print(f"!!!! MongoDB UPDATE ERROR (P{perm_id}) !!!!: {e}")
        return db

    def find_best_match(query_vector, db):
        """Performs a fast, thread-safe similarity search using the Faiss index."""
        with faiss_lock:
            if faiss_index is None or faiss_index.ntotal == 0:
                return None, -1.0
            try:
                query_vector_2d = query_vector.reshape(1, -1).astype('float32')
                D, I = faiss_index.search(query_vector_2d, k=1)
                best_similarity = D[0][0]
                if best_similarity >= MATCH_SIMILARITY_THRESHOLD:
                    matched_pid = int(I[0][0])
                    return matched_pid, best_similarity
            except Exception as e:
                print(f"!!!! FAISS SEARCH ERROR !!!!: {e}")
        return None, -1.0

    # --- Main Execution ---

    known_persons_db, next_permanent_id = load_db()
    print(f"Local cache loaded: {len(known_persons_db)} persons. next_id={next_permanent_id}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    weights = Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    detector.to(device).eval()
    transform = weights.transforms()

    tracker = BoostTrack(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=(0 if device.type == 'cuda' else device),
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
            resized_crops = [cv2.resize(c, REID_INPUT_SIZE, interpolation=cv2.INTER_LINEAR) for c in crops if
                             c.size > 0]
            if not resized_crops: return []
            tensor_batch = torch.from_numpy(np.stack(resized_crops, axis=0)).permute(0, 3, 1, 2).float().to(
                device) / 255.0
            out = rm.model(tensor_batch)
            return [l2_normalize(v.cpu().numpy()) for v in out]
        print("!!!! WARNING: Re-ID model is not callable, returning random vectors! !!!!")
        return [l2_normalize(np.random.randn(VECTOR_DIMENSION).astype(np.float32)) for _ in crops]

    def detector_worker():
        """Runs the slow detector in a separate thread."""
        print("Detector thread started...")
        while True:
            frame_to_process = frame_queue.get()
            if frame_to_process is None:
                print("Detector thread stopping...")
                break
            with torch.inference_mode():
                tensor = torch.from_numpy(cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).to(
                    torch.uint8)
                output = detector([transform(tensor).to(device)])[0]
                scores, labels, boxes = output['scores'], output['labels'], output['boxes']
                keep_mask = (scores >= 0.7) & (labels == 1)
                new_detections = np.concatenate(
                    [boxes[keep_mask].cpu().numpy(), scores[keep_mask].cpu().numpy()[:, None],
                     labels[keep_mask].cpu().numpy()[:, None]], axis=1)
                try:
                    detection_queue.get_nowait()
                except queue.Empty:
                    pass
                detection_queue.put(new_detections)

    detector_thread = threading.Thread(target=detector_worker, daemon=True)
    detector_thread.start()

    cap = cv2.VideoCapture(0)
    with torch.inference_mode():
        detections = np.empty((0, 6), dtype=np.float32)
        while True:
            success, frame = cap.read()
            if not success: break
            frame_count += 1
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            if frame_count % DETECTION_INTERVAL == 0:
                try:
                    frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass
            try:
                detections = detection_queue.get_nowait()
            except queue.Empty:
                pass
            res = tracker.update(detections, frame)

            current_frame_track_ids = set()
            for *box, track_id, conf, cls, ind in res:
                track_id = int(track_id)
                current_frame_track_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                label, color = f"T{track_id}", (0, 165, 255)
                if track_id in track_id_to_permanent_id_map:
                    pid = track_id_to_permanent_id_map[track_id]
                    label, color = f"P{pid}", (0, 255, 0)
                    if (time.time() - last_update_time_for_pid[pid]) > UPDATE_INTERVAL:
                        vecs = extract_embeddings_from_crops([crop])
                        if vecs:
                            known_persons_db = incremental_update(known_persons_db, pid, vecs[0])
                            last_update_time_for_pid[pid] = time.time()
                else:
                    vecs = extract_embeddings_from_crops([crop])
                    if not vecs: continue
                    match_id, match_sim = find_best_match(vecs[0], known_persons_db)
                    if match_id is not None:
                        track_id_to_permanent_id_map[track_id] = match_id
                        label, color = f"P{match_id} (S:{match_sim:.2f})", (0, 255, 0)
                    else:
                        buf = active_track_buffers[track_id]
                        buf["vectors"].append(vecs[0])
                        label = f"New? T{track_id} ({len(buf['vectors'])})"
                        if len(buf["vectors"]) >= MIN_ENROLL_FRAMES:
                            new_pid, known_persons_db, next_permanent_id = enroll_person(known_persons_db,
                                                                                         next_permanent_id,
                                                                                         buf["vectors"])
                            track_id_to_permanent_id_map[track_id] = new_pid
                            del active_track_buffers[track_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- cleanup ---
            stale = [tid for tid, b in active_track_buffers.items() if
                     (time.time() - b.get("last_seen", time.time())) > 5.0]
            for tid in stale: del active_track_buffers[tid]
            disappeared = [tid for tid in track_id_to_permanent_id_map if tid not in current_frame_track_ids]
            for tid in disappeared:
                pid = track_id_to_permanent_id_map.pop(tid, None)
                if pid and pid in recent_vector_buffers: del recent_vector_buffers[pid]

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("ReID Vectors (MongoDB - Faiss)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                frame_queue.put(None)
                break
            elif key == ord('r'):
                print("RESET: Deleting all entries from MongoDB...")
                try:
                    persons_collection.delete_many({})
                    faiss_collection.delete_many({})  # Also delete the index
                    print("MongoDB collections have been cleared.")
                except Exception as e:
                    print(f"!!!! MongoDB RESET ERROR !!!!: {e}")
                known_persons_db.clear()
                next_permanent_id = 1
                active_track_buffers.clear()
                track_id_to_permanent_id_map.clear()
                rebuild_faiss_index(known_persons_db)
                print("Program memory (cache and index) has been cleared.")
            elif key == ord('d'):
                with faiss_lock:
                    print("\n--- DB STATUS (d) ---")
                    print(f"Persons in cache: {len(known_persons_db)}")
                    print(f"Vectors in Faiss index: {faiss_index.ntotal if faiss_index else 0}")
                    print(f"Next available ID: {next_permanent_id}")
                    for pid, data in known_persons_db.items():
                        created_date = data.get('created', 'N/A')
                        if isinstance(created_date, datetime):
                            created_date = created_date.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"  - P{pid}: {data['count']} vectors, Created: {created_date}")
                    print("------------------\n")
            elif key == ord('k'):
                if known_persons_db:
                    try:
                        last_id = max(known_persons_db.keys())
                        persons_collection.delete_one({"permanent_id": int(last_id)})
                        print(f"DELETED (k): P{last_id} deleted from MongoDB.")
                        del known_persons_db[last_id]
                        active_track_buffers.clear()
                        track_id_to_permanent_id_map.clear()
                        rebuild_faiss_index(known_persons_db)
                        print(f"DELETED (k): P{last_id} deleted from local cache. Index rebuilt.")
                    except Exception as e:
                        print(f"Error deleting person: {e}")
                else:
                    print("DB is already empty. Nothing to delete.")

    cap.release()
    cv2.destroyAllWindows()
    mongo_client.close()
    print("Exited cleanly. DB entries (in cache):", len(known_persons_db))


if __name__ == "__main__":
    main()