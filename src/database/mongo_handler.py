# mongo_handler.py
import pymongo
import numpy as np
from typing import Optional, Dict
from utils.config import Config
from utils.logger import get_logger

logger = get_logger()

class MongoHandler:
    """
    Handles connection and basic operations with MongoDB.
    Provides collection references for other components.
    """

    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.persons_collection = None
        self._connect()

    def _connect(self):
        """
        Establishes connection to MongoDB and sets up collections.
        """
        logger.info("Connecting to MongoDB...")
        try:
            self.mongo_client = pymongo.MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
            self.mongo_client.server_info()  # Trigger connection check
            self.db = self.mongo_client[Config.DB_NAME]
            self.persons_collection = self.db[Config.COLLECTION]

            # Ensure index for faster lookups/uniqueness check
            self.persons_collection.create_index("permanent_id", unique=True)
            logger.info("Successfully connected to MongoDB and verified collections.")

        except pymongo.errors.ServerSelectionTimeoutError as e:
            logger.critical(f"MongoDB CONNECTION ERROR: {e}")
            raise RuntimeError("MongoDB connection failed. Is mongod running?") from e
        except Exception as e:
            logger.critical(f"UNEXPECTED MONGODB ERROR: {e}")
            raise RuntimeError("MongoDB initialization failed.") from e

    def close(self):
        """
        Closes the MongoDB connection gracefully.
        """
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB connection closed.")

    def get_person_by_id(self, pid: int) -> Optional[Dict]:
        """
        Fetches a single person's document from MongoDB by permanent_id.
        Supports Multi-Vector Gallery format.

        Args:
            pid (int): The permanent ID of the person.

        Returns:
            dict: The person's document or None if not found.
        """
        try:
            doc = self.persons_collection.find_one({"permanent_id": pid})
            if doc and "vectors" in doc:
                # Optimized list comprehension for conversion
                doc["vectors"] = [np.frombuffer(v, dtype=np.float32) for v in doc["vectors"]]
                
            return doc
        except Exception as e:
            logger.error(f"Error fetching person P{pid} from MongoDB: {e}")
            return None

    def load_all_persons(self) -> Dict[int, Dict]:
        """
        Loads every stored person into memory.

        Returns:
            dict: {permanent_id: {"vectors": [np.ndarray, ...], "created": datetime}}
        """
        persons = {}
        try:
            cursor = self.persons_collection.find(
                {}, {"permanent_id": 1, "vectors": 1, "created": 1, "_id": 0}
            )
            for doc in cursor:
                vectors = [np.frombuffer(v, dtype=np.float32) for v in doc.get("vectors") or []]
                if vectors:
                    persons[int(doc["permanent_id"])] = {
                        "vectors": vectors,
                        "created": doc.get("created"),
                    }
            logger.info(f"Loaded {len(persons)} persons from MongoDB.")
        except Exception as e:
            logger.error(f"Error loading persons from MongoDB: {e}")
        return persons

    def get_next_id(self) -> int:
        """
        Finds the highest permanent_id in the database and returns the next available one.
        This is a fallback for when the DB is empty or for PersonManager initialization.
        """
        try:
            # Find the document with the highest permanent_id
            last_person = self.persons_collection.find_one(sort=[("permanent_id", -1)])
            if last_person:
                return last_person['permanent_id'] + 1
            else:
                return 1  # Database is empty
        except Exception as e:
            logger.error(f"Error getting next ID from MongoDB: {e}")
            return 1
