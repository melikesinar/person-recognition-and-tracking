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
    MATCH_SIMILARITY_THRESHOLD = 0.85
    
    # --- Strict Re-Entry Logic (To prevent Imposters) ---
    REENTRY_SIMILARITY_THRESHOLD = 0.88 
    
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
    CAMERA_HOMOGRAPHIES = {
        '0-1': [[8.67880128, 45.28411211, -5167.53237445], [-2.20867665, 7.96352372, 2044.28289896], [0.04583942, 0.05740988, 1.00000000]],
        # room videos
        # '0-1': [[-0.78447575, -10.85648451, 1760.79567792], [-0.26645472, -1.73492810, 81.34685166], [-0.00401496, -0.01035841, 1.00000000]],
        # # campus videos
        # '0-1': [[3.26182811, 6.25747138, 574.22413335], [0.33626874, 1.86404089, 155.63783026], [0.00122571, 0.00351370, 1.00000000]],
        # # corridor
    }

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
