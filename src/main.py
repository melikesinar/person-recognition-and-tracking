# main_reid_vectors.py
import os
import cv2
import time
import torch
import pymongo
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
# --- YENİ İÇE AKTARMALAR ---
import threading
import queue

# --- YENİ İÇE AKTARMALAR SONU ---

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
    DETECTION_INTERVAL = 5  # Run detector every 5 frames
    REID_INPUT_SIZE = (128, 256)

    # --- YENİ: THREAD İLETİŞİM KUYRUKLARI ---
    # Ana iş parçacığından detektör iş parçacığına kareleri göndermek için
    # maxsize=1: Eğer detektör yavaşsa, eski karelerin birikmesini engeller.
    frame_queue = queue.Queue(maxsize=1)

    # Detektör iş parçacığından ana iş parçacığına sonuçları göndermek için
    detection_queue = queue.Queue(maxsize=1)
    # --- YENİ SONU ---

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

    # --- YARDIMCI FONKSİYONLAR (DEĞİŞİKLİK YOK) ---
    def load_db():
        """Loads all persons from MongoDB into a local cache (dictionary)."""
        print("Loading database from MongoDB...")
        db_cache = {}
        try:
            all_persons = list(persons_collection.find({}))
            if not all_persons:
                print("No records found in MongoDB. Starting fresh.")
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

        except Exception as e:
            print(f"Error loading DB, creating new one: {e}")
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
        """Enrolls a new person into both the local cache and MongoDB."""
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
        return next_id, db, next_id + 1

    def incremental_update(db, perm_id, new_vector):
        """
        Updates both the all-time master_vector (if not locked)
        and the recent_master_vector (rolling average).
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
        """Checks the query_vector against BOTH the master_vector and recent_master_vector."""
        best_id = None
        best_sim = -1.0
        for pid, data in db.items():
            if 'master_vector' not in data:
                continue
            master_vec = data['master_vector']
            recent_vec = data.get('recent_master_vector', master_vec)
            sim_master = cosine_similarity(query_vector, master_vec)
            sim_recent = cosine_similarity(query_vector, recent_vec)
            sim = max(sim_master, sim_recent)
            if sim > best_sim:
                best_sim = sim
                best_id = pid
        if best_sim >= MATCH_SIMILARITY_THRESHOLD:
            return best_id, best_sim
        return None, best_sim

    # --- YARDIMCI FONKSİYONLAR SONU ---

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
        vec = l2_normalize(np.random.randn(512).astype(np.float32))
        return [vec for _ in crops]

    # --- YENİ: DETEKTÖR İŞ PARÇACIĞI (THREAD) ---
    def detector_worker():
        """
        Runs the slow detector in a separate thread.
        Waits for frames from 'frame_queue', processes them,
        and puts results into 'detection_queue'.
        """
        print("Detector thread started...")
        while True:
            # frame_queue.get() bloklayıcıdır, yani yeni kare gelene kadar bekler.
            frame_to_process = frame_queue.get()
            if frame_to_process is None:  # Kapanış sinyali
                print("Detector thread stopping...")
                break

            # --- DÜZELTME: Bu thread'in kendi inference_mode'una ihtiyacı var ---
            # Ana thread'deki 'inference_mode'u miras almaz.
            with torch.inference_mode():
                # --- DÜZELTME SONU ---

                # --- 1. Detection (Yavaş Kısım) ---
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

                # Bu .numpy() çağrıları artık güvenli
                boxes_filtered = boxes[keep_mask].cpu().numpy()
                scores_filtered = scores[keep_mask].cpu().numpy()
                labels_filtered = labels[keep_mask].cpu().numpy()

                new_detections = np.concatenate(
                    [boxes_filtered, scores_filtered[:, None], labels_filtered[:, None]], axis=1)

                # --- 2. Sonuçları Gönder ---
                # Ana iş parçacığına göndermeden önce eski sonucu temizle
                try:
                    detection_queue.get_nowait()
                except queue.Empty:
                    pass  # Kuyruk zaten boşsa sorun yok

                # Yeni sonucu kuyruğa koy
                detection_queue.put(new_detections)

    # Detektör iş parçacığını başlat
    detector_thread = threading.Thread(target=detector_worker, daemon=True)
    detector_thread.start()
    # --- YENİ THREAD SONU ---

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

            # --- 1. Detektör İşlemleri (GÜNCELLENDİ) ---

            # Her 5 karede bir, o anki kareyi detektör thread'ine yolla
            if frame_count % DETECTION_INTERVAL == 0:
                try:
                    # Kopyasını yollamak daha güvenli olabilir
                    frame_copy = frame.copy()
                    # put_nowait: Eğer kuyruk doluysa (detektör yavaşsa)
                    # bekleme yapmaz, kareyi atlar ve ana akış devam eder.
                    frame_queue.put_nowait(frame_copy)
                except queue.Full:
                    # Detektör yetişemiyor demektir, sorun değil, akışa devam et
                    pass

            # Her karede, detektör thread'inden yeni sonuç gelmiş mi diye BEKLEMEDEN kontrol et
            try:
                # get_nowait: Eğer kuyrukta yeni sonuç yoksa hata verir,
                # varsa alır. Ana akışı ASLA tıkamaz.
                detections = detection_queue.get_nowait()
            except queue.Empty:
                # Yeni sonuç yok, bir önceki turun sonuçlarıyla devam et
                pass

            # --- 1. KISIM SONU --- (Yavaş 'detector()' çağrısı buradan kaldırıldı)

            # --- 2. Tracking (HER KAREDE ÇALIŞIR - HIZLI) ---
            # Takipçi, her zaman eldeki en son 'detections' setini kullanır.
            res = tracker.update(detections, frame)

            # --- 3. Re-ID ve Çizim (HER KAREDE ÇALIŞIR - HIZLI/SEYRELTİLMİŞ) ---
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
                box_color = (0, 165, 255)  # Orange (New?)

                if track_id in track_id_to_permanent_id_map:
                    # Case 1: Person is already recognized
                    pid = track_id_to_permanent_id_map[track_id]
                    label_to_display = f"P{pid}"
                    box_color = (0, 255, 0)  # Green (Recognized)

                    # Re-ID ve DB güncellemesini 1 saniyede bir yap (Seyreltme)
                    current_time_for_update = time.time()
                    if (current_time_for_update - last_update_time_for_pid[pid]) > UPDATE_INTERVAL:
                        vecs = extract_embeddings_from_crops([crop])
                        if vecs:
                            current_vector = vecs[0]
                            known_persons_db = incremental_update(known_persons_db, pid, current_vector)
                            last_update_time_for_pid[pid] = current_time_for_update

                else:
                    # Case 2: New person, try to match
                    # (Bu da seyreltilebilir, ama yeni kişi için daha sık çalıştırmak iyidir)
                    vecs = extract_embeddings_from_crops([crop])
                    if not vecs: continue
                    current_vector = vecs[0]

                    match_id, match_sim = find_best_match(current_vector, known_persons_db)
                    if match_id is not None:
                        # Match found!
                        track_id_to_permanent_id_map[track_id] = match_id
                        label_to_display = f"P{match_id} (S:{match_sim:.2f})"
                        box_color = (0, 255, 0)
                    else:
                        # Case 3: No match, start enrollment
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

                # --- 4. Draw Results ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label_to_display, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # cleanup (Tüm bu kısım hızlıdır ve ana thread'de kalır)
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
            cv2.imshow("ReID Vectors (MongoDB - Threaded)", frame)

            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                frame_queue.put(None)  # Detektör thread'ine kapanma sinyali yolla
                break

            elif key == ord('r'):
                # 'r' -> RESET MongoDB (Reset)
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
                print("Program memory (cache) has been cleared.")

            elif key == ord('d'):
                # 'd' -> Dump Database Status (Dump)
                print("\n--- DB STATUS (d) (Local Cache) ---")
                print(f"Persons in cache: {len(known_persons_db)}")
                print(f"Next available ID: {next_permanent_id}")
                for pid, data in known_persons_db.items():
                    created_date = data.get('created', 'N/A')
                    if isinstance(created_date, datetime):
                        created_date = created_date.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"  - P{pid}: {data['count']} vectors, Created: {created_date}, Master Signal: {np.sum(data['master_vector']):.4f}")
                print("------------------ Check MongoDB Compass for the full list of records.\n")

            elif key == ord('k'):
                # 'k' -> Kill (delete) the Last Added Person (Kill)
                if known_persons_db:
                    try:
                        last_id = max(known_persons_db.keys())
                        persons_collection.delete_one({"permanent_id": last_id})
                        print(f"DELETED (k): P{last_id} deleted from MongoDB.")
                        del known_persons_db[last_id]
                        active_track_buffers.clear()
                        track_id_to_permanent_id_map.clear()
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