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
    TEMP_FAISS_FILENAME = "temp_faiss.index"

    # --- Detection & Matching Thresholds ---
    DEFAULT_DETECTION_SCORE = 0.7
    DEFAULT_MATCH_THRESHOLD = 0.85

    DETECTION_SCORE_THRESHOLD = 0.65
    MATCH_SIMILARITY_THRESHOLD = 0.73

    # --- Enrollment Parameters ---
    MIN_ENROLL_FRAMES = 30
    MIN_BBOX_AREA_ENROLL = 10000
    MIN_SHARPNESS_ENROLL = 50
    MAX_VECTORS_FOR_UPDATE = 500

    # --- Buffer & Timing Settings ---
    RECENT_VECTOR_BUFFER_SIZE = 100
    DETECTION_INTERVAL = 2
    UPDATE_INTERVAL = 1.0
    BUFFER_CLEANUP_INTERVAL = 60.0
    BUFFER_MAX_IDLE_TIME = 3600.0

    # --- Re-ID Model Settings ---
    REID_INPUT_SIZE = (128, 256)
    VECTOR_DIMENSION = 512

    # ImageNet normalization (for pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # --- Tracking Parameters ---
    SHORT_TERM_MEMORY_TTL = 4.0  # seconds
    SPATIAL_PREDICTION_FACTOR_1 = 50
    SPATIAL_PREDICTION_FACTOR_2 = 100
    ADAPTIVE_THRESHOLD_FACTOR = 0.75
    ADAPTIVE_THRESHOLD_MAX = 200.0
    ADAPTIVE_THRESHOLD_MIN = 30.0
    SPATIAL_IOU_WEIGHT = 0.4
    SPATIAL_DIST_WEIGHT = 0.4
    SPATIAL_TIME_WEIGHT = 0.2
    SPATIAL_MATCH_MIN_SCORE = 0.3

    # --- Faiss Index Parameters ---
    FAISS_M = 48  # HNSW M parameter
    FAISS_EF_CONSTRUCTION = 200
    FAISS_EF_SEARCH = 64
    MIN_REBUILD_INTERVAL = 30  # seconds
    BATCH_SIZE_TRIGGER = 5  # enrollments

    # --- EMA Parameters ---
    EMA_ALPHA = 0.1

    # --- Async DB Writer ---
    ASYNC_DB_WRITER_QUEUE_SIZE = 1000

    # --- Quality Checker Settings ---
    Q_WEIGHTS = {
        'aspect': 1.5,
        'size': 1.5,
        'sharpness': 2.0,
        'edges': 1.0,
        'brightness': 1.0,
        'position': 0.5
    }
    Q_MIN_QUALITY_THRESHOLD = 0.6
    Q_MAX_ASPECT_RATIO = 4.0
    Q_MIN_ASPECT_RATIO = 0.5
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

    # --- TTA Settings ---
    USE_TTA_DEFAULT = True
    TTA_HORIZONTAL_FLIP = True
    TTA_MULTI_SCALE = False


    @staticmethod
    def validate():
        """Validates critical configuration parameters."""
        if not Config.MONGO_URI:
            raise ValueError("MONGO_URI environment variable is not set!")
        if not Config.DB_NAME:
            raise ValueError("DB_NAME environment variable is not set!")
        if not Config.COLLECTION:
            raise ValueError("COLLECTION environment variable is not set!")

        print("✓ Configuration validated successfully.")

    @staticmethod
    def print_config():
        """Prints current configuration (for debugging)."""
        print("\n--- CONFIGURATION ---")
        print(f"Database: {Config.DB_NAME}")
        print(f"Collection: {Config.COLLECTION}")
        print(f"Detection Threshold: {Config.DETECTION_SCORE_THRESHOLD}")
        print(f"Match Threshold: {Config.MATCH_SIMILARITY_THRESHOLD}")
        print(f"Min Enroll Frames: {Config.MIN_ENROLL_FRAMES}")
        print(f"TTA: {'Enabled' if Config.USE_TTA_DEFAULT else 'Disabled'}")
        print("--------------------\n")