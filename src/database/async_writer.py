"""
Asynchronous database writer for non-blocking MongoDB operations.
"""
import queue
import threading
from utils.config import Config
from utils.logger import get_logger

logger = get_logger()

class AsyncDBWriter:
    """
    Background thread that processes database write operations asynchronously.
    Prevents main thread blocking during MongoDB I/O.
    """

    def __init__(self, persons_collection, stop_event):
        """
        Initialize async writer.

        Args:
            persons_collection: MongoDB persons collection
            stop_event: Threading event for shutdown signal
        """
        self.persons_collection = persons_collection
        self.stop_event = stop_event

        self.write_queue = queue.Queue(maxsize=Config.ASYNC_DB_WRITER_QUEUE_SIZE)
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

        logger.info("Async database writer started.")

    def _safe_put(self, item: dict):
        """
        Safely puts an item into the queue without blocking the main thread.
        If the queue is full (DB overload), the item is dropped to preserve realtime performance.
        """
        try:
            self.write_queue.put(item, block=False)
        except queue.Full:
            # Drop the packet rather than freezing the video feed
            op_type = item.get("type", "unknown")
            logger.warning(f"DB Queue Full! Dropping '{op_type}' operation to maintain FPS")

    def queue_insert(self, doc: dict):
        """Queue a document for insertion."""
        self._safe_put({"type": "insert", "doc": doc})

    def queue_update(self, pid: int, update_doc: dict):
        """Queue a document update."""
        self._safe_put({"type": "update", "pid": pid, "update_doc": update_doc})

    def queue_delete(self, pid: int):
        """Queue a document deletion."""
        self._safe_put({"type": "delete", "pid": pid})

    def queue_reset(self):
        """Queue a full database reset."""
        self._safe_put({"type": "reset"})

    def flush(self):
        """
        Blocks until all currently queued operations are processed.
        Useful for ensuring consistency before critical operations.
        Note: This blocks the caller explicitly.
        """
        barrier_event = threading.Event()
        try:
            # We accept blocking here slightly just to ensure the barrier gets in
            self.write_queue.put({"type": "barrier", "event": barrier_event}, timeout=2.0)
            barrier_event.wait()
        except queue.Full:
            logger.warning("Could not flush DB queue, it is completely stuck.")

    def _writer_worker(self):
        """Background worker thread that processes queued operations."""
        logger.info("Database writer thread started...")

        while not self.stop_event.is_set():
            try:
                # Wait for a task, but timeout periodically to check stop_event
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
                elif op_type == "barrier":
                    operation["event"].set()
                else:
                    logger.error(f"Unknown DB operation type: {op_type}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[ASYNC] DB writer unexpected error: {e}")

        logger.info("Database writer thread exited.")

    def _handle_insert(self, doc):
        """Handle INSERT operation."""
        try:
            self.persons_collection.insert_one(doc)
            pid = doc["permanent_id"]
            logger.info(f"[ASYNC] MongoDB INSERT: P{pid} completed.")
        except Exception as e:
            logger.error(f"[ASYNC] MongoDB INSERT ERROR: {e}")

    def _handle_update(self, pid, update_doc):
        """Handle UPDATE operation."""
        try:
            self.persons_collection.update_one(
                {"permanent_id": int(pid)},
                update_doc
            )
        except Exception as e:
            logger.error(f"[ASYNC] MongoDB UPDATE ERROR (P{pid}): {e}")

    def _handle_delete(self, pid):
        """Handle DELETE operation."""
        try:
            self.persons_collection.delete_one({"permanent_id": int(pid)})
            logger.info(f"[ASYNC] MongoDB DELETE: P{pid} completed.")
        except Exception as e:
            logger.error(f"[ASYNC] MongoDB DELETE ERROR (P{pid}): {e}")

    def _handle_reset(self):
        """Handle RESET operation."""
        try:
            self.persons_collection.delete_many({})
            logger.warning("[ASYNC] MongoDB RESET completed.")
        except Exception as e:
            logger.critical(f"[ASYNC] MongoDB RESET ERROR: {e}")

    def shutdown(self, timeout=5.0):
        """
        Gracefully shutdown writer thread.
        """
        logger.info("Waiting for pending database writes to complete...")
        try:
            self.write_queue.put(None, block=True, timeout=1.0)
            self.writer_thread.join(timeout=timeout)
        except queue.Full:
            logger.warning("Queue full during shutdown. Forcing exit.")

        if self.writer_thread.is_alive():
            logger.warning("Database writer thread did not exit cleanly.")
            return False
        else:
            logger.info("Database writer thread exited cleanly.")
            return True


class NullDBWriter:
    """
    Drop-in replacement for AsyncDBWriter used in memory-only mode
    (no MongoDB configured or reachable). Every operation is a no-op.
    """

    def queue_insert(self, doc: dict): pass
    def queue_update(self, pid: int, update_doc: dict): pass
    def queue_delete(self, pid: int): pass
    def queue_reset(self): pass
    def flush(self): pass
    def shutdown(self, timeout=5.0): return True
