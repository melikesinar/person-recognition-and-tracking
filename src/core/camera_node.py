import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

from boxmot import BoostTrack
from utils.config import Config
from utils.visualizer import Visualizer
from utils.math_utils import check_occlusions
from tracking.buffer_manager import BufferManager
from tracking.spatial_matcher import SpatialMatcher
from tracking.identity_verifier import IdentityVerifier

class CameraNode:
    """
    Manages the tracking, ReID, and identity association pipeline for a single camera source.
    """
    def __init__(self, camera_id: int, video_source: Any, global_resources: Dict[str, Any]):
        self.camera_id = camera_id
        self.video_source = video_source
        self.resources = global_resources 
        
        print(f"[{self.__class__.__name__} {camera_id}] Initializing source: {video_source}")
        
        self.buffer_manager = BufferManager()
        self.spatial_matcher = SpatialMatcher()
        self.visualizer = Visualizer()
        self.identity_verifier = IdentityVerifier()
        
        self.tracker = BoostTrack(
            reid_weights=Path(Config.REID_MODEL_PATH),
            device=(0 if self.resources['device'].type == 'cuda' else 'cpu'),
            half=False,
            with_reid=True,
            max_age=60,
            max_obs=100
        )
        
        # Verify source availability and keep it open for main loop
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"!!!! ERROR: Could not open video source for Cam {camera_id}: {video_source}")
        
        self.frame_count = 0
        self.prev_time = time.time()
        self.fps = 0.0

    def process_frame(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """
        Main processing pipeline: Tracking -> ReID -> Association -> Visualization.
        """
        self.frame_count += 1
        
        # 1. Update Tracker & Extract Features
        tracker_results, vector_map = self._process_tracking_and_reid(detections, frame)
        
        # 2. Analyze Scene (Occlusions)
        occlusion_map = check_occlusions(tracker_results)
        
        # 3. Associate Identities
        current_track_ids = self._resolve_identities(frame, tracker_results, vector_map, occlusion_map)
        
        # 4. Periodic Verification
        if self.frame_count % Config.ID_VERIFICATION_INTERVAL == 0:
            self._verify_and_correct_identities()
            
        # 5. Cleanup
        self._perform_cleanup(current_track_ids)
        
        # 6. Visualization & Metrics
        self._update_metrics()
        self._draw_interface(frame)
        
        return frame

    def _process_tracking_and_reid(self, detections: np.ndarray, frame: np.ndarray) -> Tuple[List[Any], Dict[int, np.ndarray]]:
        """Updates tracker and extracts ReID embeddings for valid crops."""
        # BoxMot update
        res = self.tracker.update(detections, frame)
        
        tracker_results = []
        crops_to_process = []
        track_ids_for_crops = []
        h, w = frame.shape[:2]

        for *box, track_id, conf, cls, ind in res:
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            
            # Clip to frame
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2]
            tracker_results.append((box, track_id, conf, cls, ind, crop))

            if crop.size > 0:
                is_good, _, _ = self.resources['quality_checker'].assess_quality(crop, box, (h, w, 3))
                if is_good:
                    crops_to_process.append(crop)
                    track_ids_for_crops.append(track_id)

        # Batch ReID extraction
        vector_map = {}
        if crops_to_process:
            embeddings = self.resources['reid_extractor'].extract_embeddings_from_crops(crops_to_process)
            vector_map = {tid: emb for tid, emb in zip(track_ids_for_crops, embeddings)}
        
        return tracker_results, vector_map

    def _resolve_identities(self, frame: np.ndarray, tracker_results: List[Any], 
                          vector_map: Dict[int, np.ndarray], occlusion_map: Dict[int, bool]) -> Set[int]:
        """
        Maps tracker IDs to Permanent IDs (PIDs) using Persistence > Spatial > ReID > Enrollment.
        """
        current_track_ids = set()
        
        # Phase 1: Reserve PIDs for existing consistent tracks
        used_pids = self._reserve_active_identities(tracker_results)

        # Phase 2: Process each track
        for (box, track_id, conf, cls, ind, crop) in tracker_results:
            current_track_ids.add(track_id)
            
            vector = vector_map.get(track_id)
            is_occluded = occlusion_map.get(track_id, False)
            pid = self.buffer_manager.track_id_to_permanent_id_map.get(track_id)
            
            label = ""
            color = (0, 0, 0)
            
            # --- Logic Branching ---
            if pid:
                # Case A: Already Assigned
                self._handle_existing_track(pid, vector, is_occluded)
                if vector is None:
                    label, color = f"P{pid} (Low Q)", (100, 255, 100)
                else:
                    label, color = f"P{pid}", (0, 255, 0)
            else:
                # Case B: Unassigned -> Try Match or Enroll
                pid, source = self._attempt_match_or_enroll(
                    track_id, box, vector, crop, is_occluded, used_pids, frame.shape
                )
                
                if pid:
                    self.buffer_manager.track_id_to_permanent_id_map[track_id] = pid
                    used_pids.add(pid)
                    
                    if source == "spatial":
                        label, color = f"P{pid} (RA)", (0, 255, 128)
                        self.buffer_manager.lost_pid_buffer.pop(pid, None)
                    elif source == "reid":
                        label, color = f"P{pid} (ReID)", (0, 255, 0)
                    else: # enrollment
                        label, color = f"New P{pid}", (255, 200, 0)
                else:
                    # Still unknown
                    if vector is None:
                        label, color = f"T{track_id} (Low Q)", (0, 100, 155)
                    else:
                        buf_len = len(self.buffer_manager.active_track_buffers.get(track_id, {}).get("vectors", []))
                        label, color = f"T{track_id} ({buf_len})", (0, 165, 255)
            
            # Update Visuals & DB State
            if pid:
                self._update_pid_state(pid, box)
            
            if is_occluded:
                label += " (!)"
                color = (0, 0, 255)

            x1, y1, x2, y2 = map(int, box)
            self.visualizer.draw_person_bbox(frame, (x1, y1, x2, y2), label, color)
            
        return current_track_ids

    def _reserve_active_identities(self, tracker_results: List[Any]) -> Set[int]:
        """Identifies PIDs that are currently claimed by valid tracks to prevent duplicate assignment."""
        used_pids = set()
        active_track_ids = {t[1] for t in tracker_results}
        
        for tid, pid in self.buffer_manager.track_id_to_permanent_id_map.items():
            if tid in active_track_ids:
                used_pids.add(pid)
        return used_pids

    def _handle_existing_track(self, pid: int, vector: Optional[np.ndarray], is_occluded: bool):
        """Updates known person with new data if conditions are met."""
        if vector is not None:
            last_update = self.buffer_manager.last_update_time_for_pid.get(pid, 0)
            if (time.time() - last_update) > Config.UPDATE_INTERVAL and not is_occluded:
                self.resources['person_manager'].incremental_update(pid, vector, is_occluded=is_occluded)
                self.buffer_manager.last_update_time_for_pid[pid] = time.time()

    def _attempt_match_or_enroll(self, track_id: int, box: Any, vector: Optional[np.ndarray], 
                               crop: np.ndarray, is_occluded: bool, used_pids: Set[int], 
                               img_shape: Tuple[int, ...]) -> Tuple[Optional[int], Optional[str]]:
        """
        Tries to associate a track with a PID using Spatial -> ReID -> Enrollment chain.
        Returns: (pid, source_type) or (None, None)
        """
        now = time.time()
        
        # 1. Spatial Match (Fast, recovers short-term lost tracks)
        spatial_pid, _ = self.spatial_matcher.find_spatial_match(
            box, self.buffer_manager.lost_pid_buffer, now
        )
        if spatial_pid and spatial_pid not in used_pids:
            return spatial_pid, "spatial"

        # 2. ReID Match (Global search)
        if vector is not None:
            match_id, _ = self.resources['faiss_manager'].search(
                vector, self.resources['match_threshold']
            )
            if match_id and match_id not in used_pids:
                return match_id, "reid"
            
            # 3. Enrollment (New Identity)
            if not is_occluded:
                return self._process_enrollment(track_id, vector, crop, box, img_shape, used_pids)
        
        return None, None

    def _process_enrollment(self, track_id: int, vector: np.ndarray, crop: np.ndarray, 
                          box: Any, img_shape: Tuple[int, ...], used_pids: Set[int]) -> Tuple[Optional[int], Optional[str]]:
        """Accumulates evidence and enrolls a new person if quality/count thresholds are met."""
        # Quality Check
        is_good, _, q_score = self.resources['quality_checker'].assess_quality(crop, box, img_shape)
        
        buf = self.buffer_manager.active_track_buffers[track_id]
        buf["vectors"].append(vector)
        buf.setdefault("quality_scores", []).append(q_score)
        buf["last_seen"] = time.time()
        
        # Maintain rolling window
        if len(buf["vectors"]) > Config.MIN_ENROLL_FRAMES:
            buf["vectors"].pop(0)
            buf["quality_scores"].pop(0)
            
        # Check Enrollment Criteria
        if len(buf["vectors"]) >= Config.MIN_ENROLL_FRAMES:
            avg_q = np.mean(buf.get("quality_scores", [0.5]))
            
            # Relaxed quality threshold for enrollment
            if avg_q >= 0.40:
                avg_vector = np.mean(buf["vectors"], axis=0)
                avg_vector /= np.linalg.norm(avg_vector)

                # Final Safety Check: Is this person actually known but missed by single-frame ReID?
                found_pid, _ = self.resources['faiss_manager'].search(
                    avg_vector, threshold=self.resources['match_threshold']
                )
                
                final_pid = None
                if found_pid:
                    if found_pid in used_pids:
                        print(f"Enroll Conflict: T{track_id} matches active P{found_pid}. Forcing NEW ID.")
                        # ID is active elsewhere, so this must be a look-alike or duplicate. Treat as new.
                        final_pid = self.resources['person_manager'].enroll_person(buf["vectors"])
                    else:
                        print(f"Enroll Merge: T{track_id} -> P{found_pid} (accumulated evidence)")
                        final_pid = found_pid
                        self.resources['person_manager'].incremental_update(final_pid, avg_vector)
                else:
                    final_pid = self.resources['person_manager'].enroll_person(buf["vectors"])
                
                # Clear buffer after decision
                if track_id in self.buffer_manager.active_track_buffers:
                    del self.buffer_manager.active_track_buffers[track_id]
                    
                return final_pid, "enroll"
                
        return None, None

    def _update_pid_state(self, pid: int, box: Any):
        """Updates last known location and timestamp for a PID."""
        self.buffer_manager.last_known_box_for_pid[pid] = box
        
        # Sync with PersonManager if needed (e.g., first time seeing this PID on this node)
        if pid not in self.resources['person_manager'].known_persons_db:
            p_doc = self.resources['mongo_handler'].get_person_by_id(pid)
            if p_doc:
                self.resources['person_manager'].known_persons_db[pid] = p_doc
        
        if pid in self.resources['person_manager'].known_persons_db:
            self.resources['person_manager'].known_persons_db[pid]['last_seen_timestamp'] = time.time()

    def _verify_and_correct_identities(self):
        self.identity_verifier.verify_and_correct(
            self.buffer_manager,
            self.resources['faiss_manager'],
            self.resources['match_threshold'],
            self.resources['person_manager']
        )

    def _perform_cleanup(self, current_track_ids: Set[int]):
        now = time.time()
        self.buffer_manager.cleanup_stale_tracks(now)
        
        # Detect disappeared tracks
        disappeared = [
            tid for tid in self.buffer_manager.track_id_to_permanent_id_map 
            if tid not in current_track_ids
        ]
        
        for tid in disappeared:
            pid = self.buffer_manager.track_id_to_permanent_id_map.pop(tid, None)
            self.identity_verifier.clear_state_for_track(tid)
            if pid and pid in self.buffer_manager.last_known_box_for_pid:
                self.buffer_manager.add_to_lost_buffer(pid, self.buffer_manager.last_known_box_for_pid[pid], now)

        self.buffer_manager.cleanup_lost_pids(now)
        if self.buffer_manager.should_run_cleanup(now):
            self.buffer_manager.cleanup_stale_buffers(now, self.resources['person_manager'].known_persons_db)

    def _update_metrics(self):
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        if time_diff > 0.0001:
            self.fps = 1.0 / time_diff
        self.prev_time = curr_time

    def _draw_interface(self, frame: np.ndarray):
        self.visualizer.draw_hud(
            frame, self.fps, self.resources['detection_threshold'].value,
            self.resources['match_threshold'], 
            len(self.resources['person_manager'].known_persons_db), 
            self.resources['use_tta']
        )
        cv2.putText(frame, f"CAM {self.camera_id}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
