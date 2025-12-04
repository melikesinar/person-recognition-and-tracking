import multiprocessing
import queue
import numpy as np
from ultralytics import YOLO
from utils.config import Config


class DetectorWorker(multiprocessing.Process):
    """
    Manages the YOLOv8 object detection model and executes detection
    inside a dedicated background PROCESS to avoid blocking the main pipeline (GIL).
    """

    def __init__(self, frame_queue, detection_queue, detection_threshold_value, device_str):
        """
        Initializes the background detector worker.

        Args:
            frame_queue (multiprocessing.Queue): Queue for incoming frames to process.
            detection_queue (multiprocessing.Queue): Queue to push processed detection results.
            detection_threshold_value (multiprocessing.Value): Shared float for live tuning.
            device_str (str): "cuda" or "cpu" string.
        """
        super().__init__()
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.shared_threshold = detection_threshold_value
        self.device_str = device_str
        self.daemon = True

        # Pre-allocated empty detection array for fast returns
        self.EMPTY = np.zeros((0, 6), dtype=np.float32)

    def set_detection_threshold(self, threshold: float):
        """Updates the live detection threshold used during filtering."""
        self.shared_threshold.value = threshold

    def stop(self):
        """
        Signals the process to stop by sending a sentinel value into the queue,
        then waits for the process to exit cleanly.
        """
        self.frame_queue.put(None)
        self.join()

    def _clear_detection_queue(self):
        """
        Removes all pending items from the detection queue.
        Ensures only the *latest* detection result is kept.
        """
        while not self.detection_queue.empty():
            try:
                self.detection_queue.get_nowait()
            except queue.Empty:
                break

    def run(self):
        """
        Main execution loop for the worker process.
        Initializes YOLO inside the process and continuously processes frames.
        """
        print(f"Detector Process Started. Loading YOLOv8 on {self.device_str}...")
        
        try:
            # Load model INSIDE the process to avoid pickling issues
            detector = YOLO(Config.YOLO_MODEL_PATH)
            # Force device if needed, though YOLO usually auto-detects. 
            # Passing device to YOLO(...) or detector.to() can be done if strictly required.
            
            print("SUCCESS: YOLOv8 Detector Loaded in background process.")
            
            while True:
                try:
                    frame = self.frame_queue.get()
                except Exception:
                    break

                # Exit signal
                if frame is None:
                    print("Detector process stopping...")
                    break

                # Perform YOLO detection
                # verbose=False prevents stdout spam
                output = detector(frame, classes=0, verbose=False, device=self.device_str)

                if not output or output[0] is None or output[0].boxes is None:
                    new_det = self.EMPTY
                else:
                    boxes = output[0].boxes
                    det = boxes.data.cpu().numpy()

                    if det.size == 0:
                        new_det = self.EMPTY
                    else:
                        # Read threshold from shared memory
                        thresh = self.shared_threshold.value
                        new_det = det[det[:, 4] >= thresh]

                        if new_det.shape[1] > 6:
                            new_det = new_det[:, :6]

                self._clear_detection_queue()
                self.detection_queue.put(new_det)
                
        except Exception as e:
            print(f"!!!! CRITICAL ERROR in Detector Process !!!!: {e}")

