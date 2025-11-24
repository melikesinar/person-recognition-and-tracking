import torch
import queue
import threading
import numpy as np
from ultralytics import YOLO
from utils.config import Config


class DetectorWorker:
    """
    Manages the YOLOv8 object detection model and executes detection
    inside a dedicated background thread to avoid blocking the main pipeline.
    """

    def __init__(self, frame_queue, detection_queue, detection_threshold, device):
        """
        Initializes the background detector worker.

        Args:
            frame_queue (queue.Queue): Queue for incoming frames to process.
            detection_queue (queue.Queue): Queue to push processed detection results.
            detection_threshold (float): Live-tunable detection confidence threshold.
            device (torch.device): CUDA or CPU device.
        """
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.device = device

        # Live-updatable threshold (kept local for thread-safe reads)
        self._current_detection_threshold = detection_threshold

        print("Loading YOLOv8 Object Detector...")
        self.detector = YOLO(Config.YOLO_MODEL_PATH)   # No explicit .to(device) required
        print("SUCCESS: YOLOv8 Detector Loaded.")

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)

        # Pre-allocated empty detection array for fast returns
        self.EMPTY = np.zeros((0, 6), dtype=np.float32)

    def set_detection_threshold(self, threshold):
        """Updates the live detection threshold used during filtering."""
        self._current_detection_threshold = threshold

    def start(self):
        """Starts the detector worker thread."""
        self.thread.start()
        print("Detector thread started.")

    def stop(self):
        """
        Signals the thread to stop by sending a sentinel value into the queue,
        then waits for the thread to exit cleanly.
        """
        self.frame_queue.put(None)
        self.thread.join()

    def _clear_detection_queue(self):
        """
        Removes all pending items from the detection queue.
        Ensures only the *latest* detection result is kept,
        preventing queue buildup and stale results.
        """
        while not self.detection_queue.empty():
            try:
                self.detection_queue.get_nowait()
            except queue.Empty:
                break

    def _worker_loop(self):
        """
        Main execution loop for the worker thread.
        Continuously pulls frames from the input queue, performs YOLO detection,
        filters results based on the live threshold, and pushes detections
        into the result queue.
        """
        while True:
            frame = self.frame_queue.get()

            # Exit signal for thread termination
            if frame is None:
                print("Detector thread stopping...")
                break

            # Perform YOLO detection (person class only, class_id = 0)
            output = self.detector(frame, classes=0, verbose=False)

            # Basic validation of YOLO output
            if not output or output[0] is None or output[0].boxes is None:
                new_det = self.EMPTY
            else:
                boxes = output[0].boxes

                # Convert boxes to numpy: [x1, y1, x2, y2, score, class_id]
                det = boxes.data.cpu().numpy()

                if det.size == 0:
                    new_det = self.EMPTY
                else:
                    # Apply live configurable confidence threshold
                    new_det = det[det[:, 4] >= self._current_detection_threshold]

                    # Safety: ensure only 6 columns
                    if new_det.shape[1] > 6:
                        new_det = new_det[:, :6]

            # Replace previous result with the latest one
            self._clear_detection_queue()

            # Push fresh detections to output queue
            self.detection_queue.put(new_det)
