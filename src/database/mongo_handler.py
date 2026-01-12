# mongo_handler.py
import pymongo
import numpy as np
from typing import Optional, Tuple, List, Dict
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
        self.faiss_collection = None
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
            self.faiss_collection = self.db[Config.FAISS_COLLECTION]

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

    def load_vectors_for_faiss(self) -> Tuple[List[int], List[np.ndarray], int]:
        """
        Loads ALL vectors from the gallery of EVERY person to build a dense Faiss index.
        Flattens the structure: 1 Person with 5 vectors -> 5 Faiss entries pointing to same PID.

        Returns:
            tuple: (list_of_pids, list_of_vectors, next_pid)
        """
        logger.info("Loading vectors from MongoDB for Faiss index (Multi-Vector)...")
        try:
            # Projection: Retrieve only necessary fields
            # Using find() is generally faster than aggregate for simple retrieval
            cursor = self.persons_collection.find({}, {"permanent_id": 1, "vectors": 1, "_id": 0})

            pids = []
            vectors = []
            max_id = 0

            # Pre-allocating lists isn't feasible here as count is dynamic, 
            # but extending is usually efficient enough in Python.
            for doc in cursor:
                pid = doc.get('permanent_id', 0)
                if pid > max_id:
                    max_id = pid
                
                # Handle Multi-Vector Gallery
                doc_vectors = doc.get("vectors")
                if doc_vectors:
                    for v_blob in doc_vectors:
                        pids.append(pid)
                        vectors.append(np.frombuffer(v_blob, dtype=np.float32))

            next_pid = max_id + 1
            unique_identities = len(set(pids)) if pids else 0
            logger.info(f"Successfully loaded {len(vectors)} vectors (from {unique_identities} identities). Next PID: {next_pid}")
            return pids, vectors, next_pid

        except Exception as e:
            logger.error(f"Error loading vectors from MongoDB: {e}")
            return [], [], 1

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
