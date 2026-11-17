import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

from boxmot import BoostTrack
from utils.config import Config
from utils.visualizer import Visualizer
from utils.math_utils import check_occlusions, l2_normalize
from tracking.buffer_manager import BufferManager
from tracking.spatial_matcher import SpatialMatcher
from tracking.identity_verifier import IdentityVerifier
from utils.logger import get_logger
from core.appearance_cues import PartCues, extract_part_cues

logger = get_logger()

class CameraNode:
    """
    Manages the tracking, ReID, and identity association pipeline for a single camera source.
    """
    def __init__(self, camera_id: int, name: str, process_fps: float, global_resources: Dict[str, Any]):
        self.camera_id = camera_id
        self.name = name
        self.process_fps = process_fps
        self.resources = global_resources

        logger.info(f"[{self.__class__.__name__} {camera_id}] Initializing '{name}' @ {process_fps:g} FPS")

        self.buffer_manager = BufferManager()
        self.spatial_matcher = SpatialMatcher()
        self.visualizer = Visualizer()
        self.identity_verifier = IdentityVerifier()

        self.tracker = self._create_tracker(process_fps)
        self.last_embed_time: Dict[int, float] = {}  # track_id -> time of last ReID embedding
        # track_id -> time the track first appeared. Space-time rules use this, not
        # "now": waiting longer must not make an impossible match possible.
        self.track_first_seen: Dict[int, float] = {}
        # track_id -> why the system is not sure who this is (shown as "?")
        self.uncertain: Dict[int, str] = {}
        # track_id -> identities held by OTHER people visible in this camera at the
        # same time as this track. Seen together = proven different people, so the
        # track can never become one of them (even after they leave the frame).
        self.seen_with: Dict[int, Set[int]] = {}

        self.frame_count = 0
        self.prev_time = time.time()
        self.fps = 0.0
        self.last_display_frame: Optional[np.ndarray] = None

    @staticmethod
    def _create_tracker(process_fps: float) -> BoostTrack:
        """
        Creates the per-camera BoostTrack tracker in motion-only mode (no ReID network).

        Frame-to-frame association on a fixed camera works well with motion + box
        overlap (Kalman filter + IoU). Running the ReID network on every detection of
        every frame for the tracker was the single largest CPU cost, and each camera
        also loaded its own copy of the network. Appearance is now only computed by
        the shared ReidExtractor, and only when an identity decision needs it
        (see _needs_embedding).
        """
        return BoostTrack(
            reid_weights=Path(Config.REID_MODEL_PATH),
            device="cpu",
            half=False,
            with_reid=False,
            det_thresh=Config.DETECTION_SCORE_THRESHOLD,
            max_age=max(1, int(round(Config.TRACK_MAX_AGE_SECONDS * process_fps))),
            max_obs=100,
            # Camera-motion compensation (ECC) is only needed for moving cameras.
            # Fixed CCTV cameras do not move, and ECC is expensive on a CPU.
            use_ecc=False,
        )

    def _needs_embedding(self, track_id: int, now: float) -> bool:
        """
        Decides whether this track's crop should go through the ReID network this frame.

        * Track without an identity yet: every processed frame, so that a match or
          enrollment decision can be made quickly.
        * Track with an identity: at most once per REID_INTERVAL_ASSIGNED seconds,
          which is enough for gallery updates and identity-swap checks.
        """
        if track_id not in self.buffer_manager.track_id_to_permanent_id_map:
            return True
        return (now - self.last_embed_time.get(track_id, 0.0)) >= Config.REID_INTERVAL_ASSIGNED

    def process_frame(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """
        Main processing pipeline: Tracking -> ReID -> Association -> Visualization.
        """
        self.frame_count += 1

        # 1. Update Tracker & Extract Features
        tracker_results, vector_map, cue_map = self._process_tracking_and_reid(detections, frame)

        # 2. Analyze Scene (Occlusions)
        occlusion_map = check_occlusions(tracker_results)

        # 3. Associate Identities
        current_track_ids = self._resolve_identities(frame, tracker_results, vector_map, cue_map, occlusion_map)

        # 4. Periodic Verification
        if self.frame_count % Config.ID_VERIFICATION_INTERVAL == 0:
            self._verify_and_correct_identities()

        # 5. Cleanup
        self._perform_cleanup(current_track_ids)

        # 6. Visualization & Metrics
        self._update_metrics()
        self._draw_interface(frame)

        return frame

    def _process_tracking_and_reid(self, detections: np.ndarray, frame: np.ndarray
                                   ) -> Tuple[List[Any], Dict[int, np.ndarray], Dict[int, PartCues]]:
        """
        Updates the tracker, then embeds (in one batch) only the tracks that need an
        appearance vector this frame and whose crop passes the quality check. The
        same crops also get part colour cues (cheap: a few histograms).

        Returns:
            tracker_results: list of (box, track_id, conf, cls, ind, crop)
            vector_map: {track_id: embedding} for the tracks embedded this frame
            cue_map: {track_id: PartCues} for the same tracks
        """
        res = self.tracker.update(detections, frame)

        tracker_results = []
        crops_to_embed, track_ids_to_embed, cue_map = [], [], {}
        now = time.time()
        h, w = frame.shape[:2]

        for *box, track_id, conf, cls, ind in res:
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)

            # Clip to frame dimensions
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            # Invalid box check
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            tracker_results.append((box, track_id, conf, cls, ind, crop))
            self.track_first_seen.setdefault(track_id, now)

            if self._needs_embedding(track_id, now):
                is_good, _, _ = self.resources['quality_checker'].assess_quality(crop, box, (h, w, 3))
                if is_good:
                    crops_to_embed.append(crop)
                    track_ids_to_embed.append(track_id)
                    cue_map[track_id] = extract_part_cues(crop, (x1, y1, x2, y2), frame.shape)

        vector_map = {}
        if crops_to_embed:
            embeddings = self.resources['reid_extractor'].extract_embeddings_from_crops(crops_to_embed)
            for tid, emb in zip(track_ids_to_embed, embeddings):
                vector_map[tid] = emb
                self.last_embed_time[tid] = now

        return tracker_results, vector_map, cue_map

    def _resolve_identities(self, frame: np.ndarray, tracker_results: List[Any],
                          vector_map: Dict[int, np.ndarray], cue_map: Dict[int, PartCues],
                          occlusion_map: Dict[int, bool]) -> Set[int]:
        """
        Maps tracker IDs to Permanent IDs (PIDs) using Persistence > Spatial > ReID > Enrollment.
        """
        current_track_ids = set()

        # Phase 1: Reserve PIDs for existing consistent tracks
        used_pids = self._reserve_active_identities(tracker_results)
        id_map = self.buffer_manager.track_id_to_permanent_id_map
        for t in tracker_results:
            own = id_map.get(t[1])
            self.seen_with.setdefault(t[1], set()).update(p for p in used_pids if p != own)

        # Phase 2: Process each track
        for (box, track_id, conf, cls, ind, crop) in tracker_results:
            current_track_ids.add(track_id)

            vector = vector_map.get(track_id)
            cues = cue_map.get(track_id)
            is_occluded = occlusion_map.get(track_id, False)
            pid = self.buffer_manager.track_id_to_permanent_id_map.get(track_id)

            label = ""
            color = (0, 0, 0)

            # --- Logic Branching ---
            if pid:
                # Case A: Already Assigned -> Verify Validity First! (Visual Guard)
                is_valid_assignment = True

                if vector is not None:
                    person_data = self.resources['person_manager'].known_persons_db.get(pid)
                    if person_data and "vectors" in person_data:
                        # Check similarity against known gallery
                        gallery = person_data["vectors"]
                        if gallery:
                            # Stack for fast dot product
                            gallery_mat = np.stack(gallery, axis=0)
                            sims = np.dot(gallery_mat, vector)
                            max_sim = np.max(sims)

                            # If similarity drops too low, the tracker likely switched targets (ID Theft)
                            if max_sim < Config.TRACK_VALIDATION_THRESHOLD:
                                logger.warning(f"VISUAL GUARD: T{track_id} revoked from P{pid}. Sim ({max_sim:.2f}) < Threshold. Possible ID Theft/Swap.")
                                del self.buffer_manager.track_id_to_permanent_id_map[track_id]
                                pid = None # Treat as unknown
                                is_valid_assignment = False

                if is_valid_assignment:
                    self._handle_existing_track(pid, vector, cues, is_occluded)
                    if vector is None:
                        label, color = f"P{pid} (Low Q)", (100, 255, 100)
                    else:
                        label, color = f"P{pid}", (0, 255, 0)

            # Note: We use 'if not pid' here so that if the guard revoked it above, we fall into Case B immediately.
            if not pid:
                # Case B: Unassigned -> Try Match or Enroll
                pid, source = self._attempt_match_or_enroll(
                    track_id, box, vector, cues, crop, is_occluded, used_pids, frame.shape
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
                elif track_id in self.uncertain:
                    # Close call between people, or looks like someone who cannot be here
                    label, color = f"T{track_id} (?)", (0, 215, 255)
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

    def _handle_existing_track(self, pid: int, vector: Optional[np.ndarray], cues: Optional[PartCues],
                               is_occluded: bool):
        """Updates known person with new data if conditions are met."""
        if vector is not None:
            last_update = self.buffer_manager.last_update_time_for_pid.get(pid, 0)
            if (time.time() - last_update) > Config.UPDATE_INTERVAL and not is_occluded:
                self.resources['person_manager'].incremental_update(pid, vector, is_occluded=is_occluded)
                # Overlapping people mix their colours, so cues are only learned when not occluded
                self.resources['person_manager'].add_cues(pid, cues)
                self.buffer_manager.last_update_time_for_pid[pid] = time.time()

    def _attempt_match_or_enroll(self, track_id: int, box: Any, vector: Optional[np.ndarray],
                               cues: Optional[PartCues], crop: np.ndarray, is_occluded: bool, used_pids: Set[int],
                               img_shape: Tuple[int, ...]) -> Tuple[Optional[int], Optional[str]]:
        """
        Tries to associate a track with a PID using Spatial -> ReID -> Enrollment chain.
        Returns: (pid, source_type) or (None, None)
        """
        now = time.time()

        # 1. Spatial guess: someone who was lost near this spot a moment ago.
        # A location alone is not enough: in a busy room, a different person often
        # appears where someone else just disappeared (measured on EPFL: this rule
        # without an appearance check was the largest source of wrong identities).
        # So the guess is only accepted if appearance confirms it (below).
        spatial_pid = None
        if Config.SPATIAL_REACQ_ENABLED:
            spatial_pid, _ = self.spatial_matcher.find_spatial_match(
                box, self.buffer_manager.lost_pid_buffer, now
            )
        if spatial_pid in used_pids or spatial_pid in self.seen_with.get(track_id, ()):
            spatial_pid = None

        # 2. ReID Match (Global search)
        if vector is not None:
            # Stricter threshold for a single-frame match, to prevent false positives.
            # Colour cues are not trusted while the person overlaps someone else.
            query_cues = [cues] if (cues is not None and not is_occluded) else []
            matcher = self.resources['identity_matcher']
            result = matcher.match(
                vector, query_cues, Config.REENTRY_SIMILARITY_THRESHOLD,
                camera_id=self.camera_id, t=self.track_first_seen.get(track_id, now),
                unavailable=used_pids | self.seen_with.get(track_id, set()),
                foot=self._foot(box), now=now,
            )

            if spatial_pid and matcher.confirms(result, spatial_pid, Config.SPATIAL_REACQ_MIN_SCORE):
                return spatial_pid, "spatial"

            if result.pid:
                logger.info(f"[Cam {self.camera_id}] ReID match: T{track_id} -> P{result.pid} {result.describe()}")
                return result.pid, "reid"

            # 3. Enrollment (New Identity)
            if not is_occluded:
                return self._process_enrollment(track_id, vector, cues, crop, box, img_shape, used_pids)

        return None, None

    def _process_enrollment(self, track_id: int, vector: np.ndarray, cues: Optional[PartCues], crop: np.ndarray,
                          box: Any, img_shape: Tuple[int, ...], used_pids: Set[int]) -> Tuple[Optional[int], Optional[str]]:
        """Accumulates evidence and enrolls a new person if quality/count thresholds are met."""
        # Quality Check
        is_good, _, q_score = self.resources['quality_checker'].assess_quality(crop, box, img_shape)

        buf = self.buffer_manager.active_track_buffers[track_id]
        buf["vectors"].append(vector)
        buf.setdefault("quality_scores", []).append(q_score)
        buf.setdefault("cues", []).append(cues)
        buf["last_seen"] = time.time()

        # Maintain rolling window
        if len(buf["vectors"]) > Config.MIN_ENROLL_FRAMES:
            buf["vectors"].pop(0)
            buf["quality_scores"].pop(0)
            buf["cues"].pop(0)

        # Check Enrollment Criteria
        if len(buf["vectors"]) >= Config.MIN_ENROLL_FRAMES:
            avg_q = np.mean(buf.get("quality_scores", [0.5]))

            # Relaxed quality threshold for enrollment
            if avg_q >= 0.40:
                # Use robust normalization
                avg_vector = l2_normalize(np.mean(buf["vectors"], axis=0))

                buf_cues = [c for c in buf["cues"] if c is not None]
                pm = self.resources['person_manager']

                # Final Safety Check: Is this person actually known but missed by single-frame ReID?
                # Uses the averaged vector and all colour observations collected so far.
                now = time.time()
                result = self.resources['identity_matcher'].match(
                    avg_vector, buf_cues, self.resources['match_threshold'],
                    camera_id=self.camera_id, t=self.track_first_seen.get(track_id, now),
                    unavailable=used_pids | self.seen_with.get(track_id, set()),
                    foot=self._foot(box), now=now,
                )

                # Only a close call is a reason to wait. "Looks like P3 but P3 is
                # elsewhere / was seen next to this person" is a proof, not doubt.
                undecided = result.ambiguous
                if result.pid is None and undecided:
                    # "Not sure": wait for more evidence instead of guessing or creating
                    # a duplicate identity. The rolling buffer keeps the newest frames.
                    since = buf.setdefault("defer_since", now)
                    if track_id not in self.uncertain:
                        logger.info(f"[Cam {self.camera_id}] NOT SURE: T{track_id} {result.explain()}. Waiting.")
                    self.uncertain[track_id] = result.explain()
                    if now - since < Config.ENROLL_MAX_DEFER_SECONDS:
                        return None, None

                final_pid = None
                if result.pid:
                    logger.info(f"[Cam {self.camera_id}] Enroll Merge: T{track_id} -> P{result.pid} {result.describe()}")
                    final_pid = result.pid
                    pm.incremental_update(final_pid, avg_vector)
                    for c in buf_cues:
                        pm.add_cues(final_pid, c)
                else:
                    final_pid = pm.enroll_person(buf["vectors"], buf_cues)
                    if result.blocked_by is not None and not undecided:
                        logger.info(f"[Cam {self.camera_id}] NEW PERSON: T{track_id} -> P{final_pid}, "
                                    f"{result.explain()}")
                    if undecided:
                        logger.info(f"[Cam {self.camera_id}] Still not sure after {now - buf['defer_since']:.0f}s "
                                    f"({result.explain()}): T{track_id} gets new P{final_pid}")

                # Clear buffer after decision
                self.uncertain.pop(track_id, None)
                if track_id in self.buffer_manager.active_track_buffers:
                    del self.buffer_manager.active_track_buffers[track_id]

                return final_pid, "enroll"

        return None, None

    def _update_pid_state(self, pid: int, box: Any):
        """Updates last known location and timestamp for a PID."""
        self.buffer_manager.last_known_box_for_pid[pid] = box

        # All persons are loaded into memory at startup, so no DB lookup is needed here.
        if pid in self.resources['person_manager'].known_persons_db:
            self.resources['person_manager'].known_persons_db[pid]['last_seen_timestamp'] = time.time()

        presence = self.resources.get('presence')
        if presence is not None:
            presence.observe(pid, self.camera_id, time.time(), self._foot(box))

    @staticmethod
    def _foot(box) -> tuple:
        """Floor contact point of a person box: bottom centre."""
        return ((float(box[0]) + float(box[2])) / 2.0, float(box[3]))

    def _verify_and_correct_identities(self):
        self.identity_verifier.verify_and_correct(
            self.buffer_manager,
            self.resources['vector_index'],
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

        for tid in [t for t in self.last_embed_time if t not in current_track_ids]:
            del self.last_embed_time[tid]
        for d in (self.track_first_seen, self.uncertain, self.seen_with):
            for tid in [t for t in d if t not in current_track_ids]:
                del d[tid]

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
        cv2.putText(frame, f"CAM {self.camera_id}: {self.name}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)