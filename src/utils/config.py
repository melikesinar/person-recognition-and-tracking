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
    
    # --- Database Configuration ---
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME")
    COLLECTION = os.getenv("COLLECTION")
    FAISS_COLLECTION = os.getenv("FAISS_COLLECTION")
    FAISS_INDEX_KEY = "primary_person_index"

    # --- Detection & Matching Thresholds ---
    DETECTION_SCORE_THRESHOLD = 0.55
    MATCH_SIMILARITY_THRESHOLD = 0.83
    
    # --- Enrollment Parameters ---
    MIN_ENROLL_FRAMES = 30
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
    ID_SWITCH_THRESHOLD_MARGIN = 0.05
    ID_SWITCH_CONFIRMATION_THRESHOLD = 3  # Number of consistent matches required to switch ID
    ID_SWITCH_COUNTER_TTL_SECONDS = 10.0  # Reset counter if no confirmation within this duration
    ID_SWITCH_COOLDOWN = 5.0 # Seconds to lock ID after a switch
    ID_DROP_THRESHOLD = 0.55 # Revocation logic

    # Threshold for merging during enrollment (High = Strict, only merge if very sure)
    ENROLLMENT_MERGE_THRESHOLD = 0.85
    
    # Maximum age (in seconds) of an ID to be considered for Auto-Merge.
    MERGE_MAX_AGE_SECONDS = 10.0

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
        '0-1': [[-3.37740978, -13.69568251, 1636.08526081], [-0.38661487, -2.99709581, -418.04782040], [-0.02070525, -0.02173024, 1.00000000]]
        # '0-1': [[7.27268679, 10.99178752, 9.70421588], [0.83555118, 3.29873005, 83.86414115], [0.00328687, 0.00630426, 1.00000000]],
    }

    # --- Faiss Index Parameters ---
    FAISS_M = 48  # HNSW M parameter
    FAISS_EF_CONSTRUCTION = 200
    FAISS_EF_SEARCH = 64
    MIN_REBUILD_INTERVAL = 30  # seconds
    BATCH_SIZE_TRIGGER = 5  # enrollments

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
    YOLO_MODEL_PATH = str(BASE_DIR / "models" / "yolov8l.pt")
    REID_MODEL_PATH = str(BASE_DIR / "models" / "osnet_x1_0_msmt17.pt")

    # --- Video Source ---
    VIDEO_SOURCES = [
        # 0,
        # str(BASE_DIR / "videos" / "3.mp4"),
        # str(BASE_DIR / "videos" / "1.mp4")
        str(BASE_DIR / "videos" / "multi-angle" / "4p-c0.avi"),
        str(BASE_DIR / "videos" / "multi-angle" / "4p-c1.avi"),
        # str(BASE_DIR / "videos" / "multi-angle" / "4p-c2.avi"),
        # str(BASE_DIR / "videos" / "multi-angle" / "4p-c3.avi")
    ]

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
        if not Config.MONGO_URI:
            raise ValueError("MONGO_URI environment variable is not set!")
        if not Config.DB_NAME:
            raise ValueError("DB_NAME environment variable is not set!")
        if not Config.COLLECTION:
            raise ValueError("COLLECTION environment variable is not set!")
        if not Config.FAISS_COLLECTION:
            raise ValueError("FAISS_COLLECTION environment variable is not set!")

    @staticmethod
    def print_config():
        """Prints current configuration (for debugging)."""
        # Deferred import to avoid circular dependency
        from utils.logger import get_logger
        logger = get_logger()

        logger.info(f"Database: {Config.DB_NAME}")
        logger.info(f"Collection: {Config.COLLECTION}")
        logger.info(f"Detection Threshold: {Config.DETECTION_SCORE_THRESHOLD}")
        logger.info(f"Match Threshold: {Config.MATCH_SIMILARITY_THRESHOLD}")
        logger.info(f"Min Enroll Frames: {Config.MIN_ENROLL_FRAMES}")
        logger.info(f"TTA: {'Enabled' if Config.USE_TTA_DEFAULT else 'Disabled'}")
