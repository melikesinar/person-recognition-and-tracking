"""
Configuration management for the Person Re-ID system.
Handles environment variables and default settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration class for all system parameters."""

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

    DETECTION_SCORE_THRESHOLD = float(os.getenv("DETECTION_SCORE_THRESHOLD", DEFAULT_DETECTION_SCORE))
    MATCH_SIMILARITY_THRESHOLD = float(os.getenv("MATCH_SIMILARITY_THRESHOLD", DEFAULT_MATCH_THRESHOLD))

    # --- Enrollment Parameters ---
    MIN_ENROLL_FRAMES = int(os.getenv("MIN_ENROLL_FRAMES", "30"))
    MIN_BBOX_AREA_ENROLL = int(os.getenv("MIN_BBOX_AREA_ENROLL", "10000"))
    MIN_SHARPNESS_ENROLL = float(os.getenv("MIN_SHARPNESS_ENROLL", "50.0"))
    MAX_VECTORS_FOR_UPDATE = int(os.getenv("MAX_VECTORS_FOR_UPDATE", "300"))

    # --- Buffer & Timing Settings ---
    RECENT_VECTOR_BUFFER_SIZE = int(os.getenv("RECENT_VECTOR_BUFFER_SIZE", "50"))
    DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL", "2"))
    UPDATE_INTERVAL = 1.0  # seconds
    BUFFER_CLEANUP_INTERVAL = 60.0  # seconds
    BUFFER_MAX_IDLE_TIME = 3600.0  # seconds

    # --- Re-ID Model Settings ---
    REID_INPUT_SIZE = (128, 256)
    VECTOR_DIMENSION = 512

    # ImageNet normalization (for pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # --- Tracking Parameters ---
    SHORT_TERM_MEMORY_TTL = 4.0  # seconds
    SPATIAL_MATCH_THRESHOLD = 150  # deprecated (now adaptive)

    # --- Faiss Index Parameters ---
    FAISS_M = 48  # HNSW M parameter
    FAISS_EF_CONSTRUCTION = 200
    FAISS_EF_SEARCH = 64
    MIN_REBUILD_INTERVAL = 30  # seconds
    BATCH_SIZE_TRIGGER = 5  # enrollments

    # --- Model Paths ---
    YOLO_MODEL_PATH = 'yolov8l.pt'
    REID_MODEL_PATH = 'osnet_x1_0_msmt17.pt'

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