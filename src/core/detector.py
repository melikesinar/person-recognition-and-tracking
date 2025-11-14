# detector.py
import torch
import queue
import threading
import numpy as np
from ultralytics import YOLO
from utils.config import Config


class DetectorWorker:
    """
    Manages the YOLOv8 object detection model and runs the detection
    in a dedicated background thread for non-blocking I/O.
    """

    def __init__(self, frame_queue, detection_queue, detection_threshold, device):
        """
        Initializes the detector worker.

        Args:
            frame_queue (queue.Queue): Input queue for frames to process.
            detection_queue (queue.Queue): Output queue for detection results.
            detection_threshold (float): Live-tuning detection score threshold.
            device (torch.device): CUDA or CPU device.
        """
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.device = device

        # Live-tuning threshold reference (can be updated externally if passed as mutable object)
        # We will use a simple attribute here for simplicity.
        self._current_detection_threshold = detection_threshold

        print("Loading Object Detector (YOLOv8)...")
        self.detector = YOLO(Config.YOLO_MODEL_PATH)
        self.detector.to(self.device)
        print("SUCCESS: YOLOv8 Detector LOADED.")

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)

    def set_detection_threshold(self, threshold):
        """Updates the live-tuning threshold."""
        self._current_detection_threshold = threshold

    def start(self):
        """Starts the worker thread."""
        self.thread.start()
        print("Detector thread (YOLOv8) started...")

    def stop(self):
        """Signals the worker thread to stop."""
        self.frame_queue.put(None)

    def _worker_loop(self):
        """
        The main loop for the detector thread.
        Pulls frames, performs detection, and pushes results.
        """
        while True:
            frame_to_process = self.frame_queue.get()
            if frame_to_process is None:  # Exit signal
                print("Detector thread stopping...")
                break

            # Run YOLO detection for 'person' class (class_id=0)
            output = self.detector(frame_to_process, classes=0, verbose=False)

            new_detections = np.empty((0, 6), dtype=np.float32)

            if output:
                boxes = output[0].boxes

                # Filter detections based on the current live-tuning threshold
                scores = boxes.conf.cpu().numpy()
                # Use the locally stored threshold value
                keep_mask = (scores >= self._current_detection_threshold)

                if keep_mask.any():
                    # Format as BoostTrack expects: [x1, y1, x2, y2, score, class_id]
                    new_detections = boxes.data[keep_mask].cpu().numpy()

                    # Safety check: limit columns to 6
                    if new_detections.shape[1] > 6:
                        new_detections = new_detections[:, :6]

            # Atomically put new detections into the queue
            try:
                # Clear the queue first if possible to ensure we send the latest result
                self.detection_queue.get_nowait()
            except queue.Empty:
                pass
            self.detection_queue.put(new_detections)