# mongo_handler.py
import pymongo
import numpy as np
from utils.config import Config


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
        print("Connecting to MongoDB...")
        try:
            self.mongo_client = pymongo.MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
            self.mongo_client.server_info()  # Try to connect
            self.db = self.mongo_client[Config.DB_NAME]
            self.persons_collection = self.db[Config.COLLECTION]
            self.faiss_collection = self.db[Config.FAISS_COLLECTION]

            # Ensure index for faster lookups/uniqueness check
            self.persons_collection.create_index("permanent_id", unique=True)
            print("Successfully connected to MongoDB and verified collections.")

        except pymongo.errors.ServerSelectionTimeoutError as e:
            print(f"!!!! MongoDB CONNECTION ERROR !!!!: {e}")
            raise RuntimeError("MongoDB connection failed.") from e
        except Exception as e:
            print(f"!!!! UNEXPECTED MONGODB ERROR !!!!: {e}")
            raise RuntimeError("MongoDB initialization failed.") from e

    def close(self):
        """
        Closes the MongoDB connection gracefully.
        """
        if self.mongo_client:
            self.mongo_client.close()
            print("MongoDB connection closed.")

    def load_db_to_cache(self, known_persons_db):
        """
        Loads all person documents from MongoDB into the application's in-memory cache.

        Args:
            known_persons_db (dict): The application's local cache dictionary.

        Returns:
            int: The next permanent ID to use.
        """
        print("Loading person records from MongoDB into cache...")
        try:
            all_persons = list(self.persons_collection.find({}))
            if all_persons:
                for doc in all_persons:
                    pid = doc['permanent_id']
                    # Use a shallow copy or dict comprehension to avoid modifying the doc directly if needed later
                    known_persons_db[pid] = {
                        "master_vector": np.frombuffer(doc['master_vector'], dtype=np.float32),
                        "count": doc['count'],
                        "created": doc['created']
                    }

                next_id = max(known_persons_db.keys()) + 1
                print(f"Successfully loaded {len(known_persons_db)} persons. Next ID: {next_id}")
                return next_id
            else:
                print("No person records found in MongoDB. Starting fresh.")
                return 1
        except Exception as e:
            print(f"Error loading DB: {e}")
            # If loading fails, start fresh
            known_persons_db.clear()
            return 1