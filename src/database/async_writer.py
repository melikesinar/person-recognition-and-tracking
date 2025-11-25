"""
Asynchronous database writer for non-blocking MongoDB operations.
"""
import queue
import threading
from bson.binary import Binary
from utils.config import Config


class AsyncDBWriter:
    """
    Background thread that processes database write operations asynchronously.
    Prevents main thread blocking during MongoDB I/O.
    """

    def __init__(self, persons_collection, faiss_collection, stop_event):
        """
        Initialize async writer.

        Args:
            persons_collection: MongoDB persons collection
            faiss_collection: MongoDB Faiss index collection
            stop_event: Threading event for shutdown signal
        """
        self.persons_collection = persons_collection
        self.faiss_collection = faiss_collection
        self.stop_event = stop_event

        self.write_queue = queue.Queue(maxsize=Config.ASYNC_DB_WRITER_QUEUE_SIZE)
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

        print("Async database writer started.")

    def queue_insert(self, doc):
        """
        Queue a document for insertion.

        Args:
            doc (dict): Document to insert
        """
        self.write_queue.put({"type": "insert", "doc": doc})

    def queue_update(self, pid, update_doc):
        """
        Queue a document update.

        Args:
            pid (int): Permanent ID
            update_doc (dict): MongoDB update document
        """
        self.write_queue.put({"type": "update", "pid": pid, "update_doc": update_doc})

    def queue_delete(self, pid):
        """
        Queue a document deletion.

        Args:
            pid (int): Permanent ID to delete
        """
        self.write_queue.put({"type": "delete", "pid": pid})

    def queue_reset(self):
        """Queue a full database reset."""
        self.write_queue.put({"type": "reset"})

    def _writer_worker(self):
        """Background worker thread that processes queued operations."""
        print("Database writer thread started...")

        while not self.stop_event.is_set():
            try:
                operation = self.write_queue.get(timeout=1.0)

                if operation is None:  # Poison pill
                    break

                op_type = operation.get("type")

                if op_type == "insert":
                    self._handle_insert(operation["doc"])
                elif op_type == "update":
                    self._handle_update(operation["pid"], operation["update_doc"])
                elif op_type == "delete":
                    self._handle_delete(operation["pid"])
                elif op_type == "reset":
                    self._handle_reset()
                else:
                    print(f"!!!! Unknown DB operation type: {op_type} !!!!")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"!!!! [ASYNC] DB writer unexpected error !!!!: {e}")

        print("Database writer thread exited.")

    def _handle_insert(self, doc):
        """Handle INSERT operation."""
        try:
            self.persons_collection.insert_one(doc)
            pid = doc["permanent_id"]
            print(f"[ASYNC] MongoDB INSERT: P{pid} completed.")
        except Exception as e:
            print(f"!!!! [ASYNC] MongoDB INSERT ERROR !!!!: {e}")

    def _handle_update(self, pid, update_doc):
        """Handle UPDATE operation."""
        try:
            self.persons_collection.update_one(
                {"permanent_id": int(pid)},
                update_doc
            )
            # Silent (too frequent to log)
        except Exception as e:
            print(f"!!!! [ASYNC] MongoDB UPDATE ERROR (P{pid}) !!!!: {e}")

    def _handle_delete(self, pid):
        """Handle DELETE operation."""
        try:
            self.persons_collection.delete_one({"permanent_id": int(pid)})
            print(f"[ASYNC] MongoDB DELETE: P{pid} completed.")
        except Exception as e:
            print(f"!!!! [ASYNC] MongoDB DELETE ERROR (P{pid}) !!!!: {e}")

    def _handle_reset(self):
        """Handle RESET operation."""
        try:
            self.persons_collection.delete_many({})
            self.faiss_collection.delete_many({})
            print("[ASYNC] MongoDB RESET completed.")
        except Exception as e:
            print(f"!!!! [ASYNC] MongoDB RESET ERROR !!!!: {e}")

    def shutdown(self, timeout=5.0):
        """
        Gracefully shutdown writer thread.

        Args:
            timeout (float): Maximum seconds to wait for pending writes

        Returns:
            bool: True if shutdown cleanly
        """
        print("Waiting for pending database writes to complete...")
        self.write_queue.put(None)  # Poison pill
        self.writer_thread.join(timeout=timeout)

        if self.writer_thread.is_alive():
            print("WARNING: Database writer thread did not exit cleanly.")
            return False
        else:
            print("Database writer thread exited cleanly.")
            return True