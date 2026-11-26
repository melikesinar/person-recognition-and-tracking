"""
Configuration management for the Person Re-ID system.
Handles environment variables and default settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class for all system parameters."""

    # --- Path Configuration ---
    # Resolve project root: src/utils/config.py -> src/utils -> src -> root
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # --- Database Configuration (optional) ---
    # If MONGO_URI is empty, or MongoDB is unreachable, the system runs in
    # memory-only mode: identities live for the session and nothing is persisted.
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME")
    COLLECTION = os.getenv("COLLECTION")
    USE_DATABASE = bool(MONGO_URI)

    # --- Detection & Matching Thresholds ---
    DETECTION_SCORE_THRESHOLD = 0.55
    # Both thresholds below apply to the FUSED score (OSNet + clothing colours,
    # see core/appearance_cues.fuse_scores), not to raw OSNet similarity.
    # Chosen at ~0.5-1% false-match rate on EPFL pairs (docs/FAZ1_RAPOR.md).
    # Enrollment safety check, using ~1.7 s of accumulated evidence:
    MATCH_SIMILARITY_THRESHOLD = 0.82

    # --- Strict Re-Entry Logic (To prevent Imposters) ---
    # A new track matched from a single observation needs a stricter score:
    REENTRY_SIMILARITY_THRESHOLD = 0.84
    
    # --- Visual Guard (Identity Theft Prevention) ---
    # If a tracked person suddenly stops looking like themselves (e.g. crossing another person),
    # drop the ID immediately.
    TRACK_VALIDATION_THRESHOLD = 0.60
    
    # --- Enrollment Parameters ---
    # Number of good-quality appearance samples needed before a new identity is created.
    # (Was 30 when every camera ran at ~30 FPS; at 6 FPS processing, 10 ≈ 1.7 seconds.)
    MIN_ENROLL_FRAMES = 10
    MIN_BBOX_AREA_ENROLL = 5000
    MIN_SHARPNESS_ENROLL = 35
    
    # --- Multi-Vector Gallery Settings ---
    GALLERY_SIZE = 60  # Max number of diverse vectors per person (Front, Back, Side, etc.)
    # Only add to gallery if similarity to existing vectors is LESS than this.
    # If a new vector is > 0.90 similar to an existing one, we assume we already have that "pose".
    GALLERY_UPDATE_THRESHOLD = 0.90

    # --- Buffer & Timing Settings ---
    UPDATE_INTERVAL = 1.0
    BUFFER_CLEANUP_INTERVAL = 60.0
    BUFFER_MAX_IDLE_TIME = 3600.0

    # --- Re-ID Model Settings ---
    REID_INPUT_SIZE = (128, 256)
    VECTOR_DIMENSION = 512

    # --- Identity Verification / Self-Correction ---
    ID_VERIFICATION_INTERVAL = 2  # Check every 2 frames
    ID_SWITCH_THRESHOLD_MARGIN = 0.04
    ID_SWITCH_CONFIRMATION_THRESHOLD = 3  # Number of consistent matches required to switch ID
    ID_SWITCH_COUNTER_TTL_SECONDS = 10.0  # Reset counter if no confirmation within this duration
    ID_SWITCH_COOLDOWN = 5.0 # Seconds to lock ID after a switch
    ID_DROP_THRESHOLD = 0.55 # Revocation logic

    # Maximum age (in seconds) of an ID to be considered for Auto-Merge.
    MERGE_MAX_AGE_SECONDS = 20.0

    # ImageNet normalization (for pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # --- Tracking Parameters ---
    SHORT_TERM_MEMORY_TTL = 30  # Remember for 30 seconds
    SPATIAL_PREDICTION_FACTOR_1 = 50
    SPATIAL_PREDICTION_FACTOR_2 = 100
    ADAPTIVE_THRESHOLD_FACTOR = 0.75
    ADAPTIVE_THRESHOLD_MAX = 200.0
    ADAPTIVE_THRESHOLD_MIN = 30.0
    SPATIAL_IOU_WEIGHT = 0.4
    SPATIAL_DIST_WEIGHT = 0.4
    SPATIAL_TIME_WEIGHT = 0.2
    SPATIAL_MATCH_MIN_SCORE = 0.2

    # --- Cross-Camera Spatial Matching (Homography) ---
    SPATIAL_MERGE_ENABLED = True
    # Maximum pixel distance between "projected point" and "actual point" to consider a spatial match
    SPATIAL_MERGE_DISTANCE_THRESHOLD = 100.0 
    # Weight of spatial score vs visual ReID score (e.g., 0.3 means 30% spatial, 70% visual)
    # If spatial match is perfect, it adds a boost.
    SPATIAL_MERGE_SCORE_WEIGHT = 0.3
    
    # Homography Matrices: Perspective transform between cameras.
    # Key format: "source_cam_id-target_cam_id" (e.g., "0-1")
    # Value: 3x3 List of Lists [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]]
    # Use a tool like 'opencv_calibration' to find these. 
    # Below is a PLACEHOLDER example (Identity matrix = no transform).
    # Default: none. Each camera file (config/cameras*.yaml) carries the matrices
    # for its own installation under a `homographies:` key; see
    # config/cameras.epfl-room.yaml and tools/calibrate_homography.py.
    CAMERA_HOMOGRAPHIES = {}

    # --- Vector Search ---
    # Number of nearest gallery vectors that vote in a search (k-NN voting).
    SEARCH_K_NEIGHBORS = 5

    # --- Async DB Writer ---

    ASYNC_DB_WRITER_QUEUE_SIZE = 100

    # --- Quality Checker Settings ---
    Q_WEIGHTS = {
        'aspect': 1.5,
        'size': 1.5,
        'sharpness': 2.0,
        'edges': 1.0,
        'brightness': 1.0,
        'position': 0.5
    }
    Q_MIN_QUALITY_THRESHOLD = 0.25
    Q_MAX_ASPECT_RATIO = 5.0
    Q_MIN_ASPECT_RATIO = 0.3
    Q_ASPECT_RATIO_OPTIMAL_MIN = 1.5
    Q_ASPECT_RATIO_OPTIMAL_MAX = 3.0
    Q_BBOX_AREA_EXCELLENT = 50000.0
    Q_SHARPNESS_EXCELLENT = 200.0
    Q_EDGE_DENSITY_MIN = 0.02
    Q_EDGE_DENSITY_MAX = 0.30
    Q_EDGE_DENSITY_OPTIMAL_MIN = 0.05
    Q_EDGE_DENSITY_OPTIMAL_MAX = 0.20
    Q_BRIGHTNESS_STD_MIN = 15
    Q_BRIGHTNESS_MEAN_MIN = 30
    Q_BRIGHTNESS_MEAN_MAX = 225
    Q_FRAME_EDGE_MARGIN = 0.05

    # --- Model Paths ---
    # Use dynamic paths based on project root
    YOLO_MODEL_PATH = str(BASE_DIR / "models" / "yolov8n.pt")
    REID_MODEL_PATH = str(BASE_DIR / "models" / "osnet_x1_0_msmt17.pt")

    # --- Cameras ---
    # Cameras are defined in a YAML file so that any number of them can be added
    # without touching code. See config/cameras.yaml for the format.
    CAMERAS_CONFIG_PATH = Path(os.getenv("CAMERAS_CONFIG", BASE_DIR / "config" / "cameras.yaml"))
    # How many frames per second each camera is processed at. A CPU cannot run
    # detection + ReID at 30 FPS for several cameras; 5-8 FPS is enough to follow walking people.
    DEFAULT_PROCESS_FPS = 6.0

    # --- Tracker ---
    # A lost track is kept alive for this long (converted to frames using the processing FPS).
    TRACK_MAX_AGE_SECONDS = 4.0

    # --- ReID preprocessing ---
    # Contrast equalisation (CLAHE) on the crop before ReID.
    REID_USE_CLAHE = True
    # A track that already has an identity is re-embedded at most this often (seconds).
    # Tracks without an identity are embedded on every processed frame.
    REID_INTERVAL_ASSIGNED = 1.0
    # On CPU, compile the ReID network with TorchScript (≈2x faster, identical output).
    REID_JIT_OPTIMIZE = True

    # --- Multi-cue identity (Faz 1) ---
    # Weight of each cue in the fused similarity. A cue that is not visible in
    # an observation drops out and the remaining weights are renormalised.
    # Measured on EPFL (docs/FAZ1_RAPOR.md): lower-body colour is almost as
    # discriminative as OSNet; the head region is close to chance at low resolution,
    # so it is off by default (raise it for high-resolution cameras).
    CUE_WEIGHTS = {"reid": 0.50, "upper": 0.20, "lower": 0.30, "head": 0.0}
    # Reject a match when BOTH top and bottom clothing colour similarities are
    # below this, however similar the OSNet vectors are.
    COLOR_VETO_THRESHOLD = 0.35
    # Colour observations kept per person (rolling window, memory only).
    CUE_GALLERY_SIZE = 30
    # Candidates shortlisted by OSNet before colour re-scoring, and the minimum
    # OSNet similarity to be shortlisted at all.
    MATCH_CANDIDATES = 5
    CANDIDATE_MIN_REID_SIM = 0.55

    # --- Space-time rules & "not sure" (Faz 2) ---
    # A match must beat every other plausible person by at least this much (fused
    # scale); otherwise the system says "not sure" and waits for more evidence.
    AMBIGUITY_MARGIN = 0.03
    # While a new track is "not sure" (or looks like someone who cannot be here),
    # creating a new identity is postponed for up to this many seconds.
    ENROLL_MAX_DEFER_SECONDS = 6.0
    # Sightings in two non-overlapping cameras closer than this (s) count as "at
    # the same time" (impossible for one person).
    SIMULTANEOUS_WINDOW_SECONDS = 1.0
    # Overlapping cameras with a homography: a person visible in both must stand on
    # the same floor spot. Max distance (px, after mapping) still counted as "same".
    SPATIAL_GATE_PX = 60.0
    # Short-term re-acquisition by location (someone lost near this spot a moment
    # ago) is only accepted if appearance also agrees: fused score at least this,
    # and no other possible person looks clearly more similar. Lower than the
    # re-entry threshold because the location is evidence too.
    SPATIAL_REACQ_MIN_SCORE = 0.70
    # Location-based re-acquisition on/off. Off by default: on EPFL (4 similar-looking
    # people) it was the largest source of wrong identities, even with the
    # appearance check (cross-camera consistency 78% off vs 57% on; FAZ2_RAPOR.md).
    SPATIAL_REACQ_ENABLED = False

    # --- Identity consolidation (Faz 3) ---
    # Periodically merge identities that are really one person split in several
    # numbers (see src/tracking/consolidator.py).
    CONSOLIDATION_ENABLED = True
    CONSOLIDATE_INTERVAL_SECONDS = 5.0
    # Whole-gallery fused similarity needed to merge (robust average, so lower
    # than the single-frame matching thresholds).
    CONSOLIDATE_MIN_SCORE = 0.80
    # Identities need at least this many gallery vectors before they are compared.
    CONSOLIDATE_MIN_VECTORS = 3
    CONSOLIDATE_MAX_MERGES = 1
    # The older auto-merge (OSNet-only, active identities younger than
    # MERGE_MAX_AGE_SECONDS, plus homography "spatial dominance").
    LEGACY_AUTO_MERGE_ENABLED = True
    # Unseen in a camera for longer than this (s) = the visit to that camera ended.
    VISIT_GAP_SECONDS = 5.0
    # Routes / dwell times are written here as CSV when the program exits.
    OUTPUT_DIR = BASE_DIR / "output"
    # Camera topology (overlaps, walking times). Set from the cameras YAML file.
    CAMERA_TOPOLOGY = {}

    # --- Detector ---
    # YOLO input size. Ultralytics' default (640) upscales small camera frames,
    # which costs CPU time without finding more people. Use 640 for high-resolution
    # cameras where people appear small.
    DETECTION_IMG_SIZE = 416

    # --- CPU threading ---
    # Detector and ReID run in separate processes and compete for the same cores.
    # Many-thread PyTorch is *slower* on small batches (measured), so keep these modest.
    DETECTOR_NUM_THREADS = 4
    REID_NUM_THREADS = 4

    # --- Logging Configuration ---
    import logging
    LOG_LEVEL = logging.DEBUG # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_TO_FILE = True
    LOG_DIR = BASE_DIR / "logs"
    LOG_FILE_NAME = "system_events.log"

    # --- TTA Settings ---
    USE_TTA_DEFAULT = False
    TTA_HORIZONTAL_FLIP = True


    @staticmethod
    def validate():
        """Validates critical configuration parameters."""
        if Config.USE_DATABASE:
            if not Config.DB_NAME:
                raise ValueError("MONGO_URI is set but DB_NAME is not!")
            if not Config.COLLECTION:
                raise ValueError("MONGO_URI is set but COLLECTION is not!")

    @staticmethod
    def load_cameras() -> list:
        """
        Reads the camera list from CAMERAS_CONFIG_PATH.

        Returns:
            list of dicts: {"id": int, "name": str, "source": int|str, "process_fps": float}
        """
        import yaml

        path = Config.CAMERAS_CONFIG_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Camera config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        default_fps = float(data.get("process_fps", Config.DEFAULT_PROCESS_FPS))
        cameras = []
        seen_ids = set()
        for i, cam in enumerate(data.get("cameras") or []):
            cam_id = int(cam.get("id", i))
            if cam_id in seen_ids:
                raise ValueError(f"Duplicate camera id {cam_id} in {path}")
            seen_ids.add(cam_id)

            source = cam["source"]
            if isinstance(source, str):
                if source.isdigit():
                    source = int(source)  # webcam index given as a string
                elif "://" not in source:
                    # Local video file: resolve relative paths from the project root
                    p = Path(source)
                    source = str(p if p.is_absolute() else Config.BASE_DIR / p)

            cameras.append({
                "id": cam_id,
                "name": str(cam.get("name", f"cam{cam_id}")),
                "source": source,
                "process_fps": float(cam.get("process_fps", default_fps)),
            })

        if not cameras:
            raise ValueError(f"No cameras defined in {path}")

        # Homographies belong to a specific camera installation, so they live in the
        # same file as the cameras. A matrix from another installation would produce
        # confident but wrong "same location" matches.
        homographies = data.get("homographies") or {}
        Config.CAMERA_HOMOGRAPHIES = {str(k): v for k, v in homographies.items()}
        Config.CAMERA_TOPOLOGY = data.get("topology") or {}
        return cameras

    @staticmethod
    def print_config():
        """Prints current configuration (for debugging)."""
        # Deferred import to avoid circular dependency
        from utils.logger import get_logger
        logger = get_logger()

        if Config.USE_DATABASE:
            logger.info(f"Database: {Config.DB_NAME} / {Config.COLLECTION}")
        else:
            logger.info("Database: disabled (memory-only mode)")
        logger.info(f"Detection Threshold: {Config.DETECTION_SCORE_THRESHOLD}")
        logger.info(f"Match Threshold: {Config.MATCH_SIMILARITY_THRESHOLD}")
        logger.info(f"Min Enroll Frames: {Config.MIN_ENROLL_FRAMES}")
        logger.info(f"TTA: {'Enabled' if Config.USE_TTA_DEFAULT else 'Disabled'}")
