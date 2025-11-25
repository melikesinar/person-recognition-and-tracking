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

    def get_person_by_id(self, pid):
        """
        Fetches a single person's document from MongoDB by permanent_id.

        Args:
            pid (int): The permanent ID of the person.

        Returns:
            dict: The person's document or None if not found.
        """
        try:
            doc = self.persons_collection.find_one({"permanent_id": pid})
            if doc:
                # Convert BSON binary back to numpy array
                doc["master_vector"] = np.frombuffer(doc['master_vector'], dtype=np.float32)
            return doc
        except Exception as e:
            print(f"!!!! Error fetching person P{pid} from MongoDB !!!!: {e}")
            return None

    def load_vectors_for_faiss(self):
        """
        Loads only the necessary data (ID and vector) from MongoDB
        to build the Faiss index, without loading the entire database into memory.

        Returns:
            tuple: (list_of_pids, list_of_vectors, next_pid)
        """
        print("Loading vectors from MongoDB for Faiss index...")
        try:
            # Projection: only retrieve the fields we absolutely need
            cursor = self.persons_collection.find({}, {"permanent_id": 1, "master_vector": 1, "_id": 0})

            pids = []
            vectors = []
            max_id = 0

            for doc in cursor:
                pid = doc['permanent_id']
                pids.append(pid)
                vectors.append(np.frombuffer(doc['master_vector'], dtype=np.float32))
                if pid > max_id:
                    max_id = pid

            next_pid = max_id + 1
            print(f"Successfully loaded {len(pids)} vectors. Next available PID will be {next_pid}")
            return pids, vectors, next_pid

        except Exception as e:
            print(f"!!!! Error loading vectors from MongoDB !!!!: {e}")
            return [], [], 1

    def get_next_id(self):
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
            print(f"!!!! Error getting next ID from MongoDB !!!!: {e}")
            return 1