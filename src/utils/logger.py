import logging
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from utils.config import Config

class Logger:
    _instance = None

    @staticmethod
    def get_logger(name="PRT_System"):
        if Logger._instance is None:
            Logger._instance = Logger._setup_logger(name)
        return Logger._instance

    @staticmethod
    def _setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(Config.LOG_LEVEL)
        logger.propagate = False  # Prevent double logging if attached to root

        if logger.handlers:
            return logger

        # Formatter
        # Example: 2023-10-27 15:30:01 [INFO] [mongo_handler.py:45] - Connecting to DB...
        log_format = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. Console Handler (Stream)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(Config.LOG_LEVEL)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        # 2. File Handler (Rotating)
        if Config.LOG_TO_FILE:
            log_dir = Config.LOG_DIR
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename with timestamp
            base_name = Config.LOG_FILE_NAME
            if "." in base_name:
                stem, ext = base_name.rsplit(".", 1)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{stem}_{timestamp}.{ext}"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{base_name}_{timestamp}.log"

            file_path = log_dir / new_name
            
            # Max 5MB per file, keep last 3 backups
            file_handler = RotatingFileHandler(
                file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
            )
            file_handler.setLevel(Config.LOG_LEVEL)
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)

        return logger

# Convenience function
def get_logger(name="PRT_System"):
    return Logger.get_logger(name)
