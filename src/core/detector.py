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

    def __init__(self, frame_queue, detection_queue, detection_threshold_value, device_str, ready_event):
        """
        Initializes the background detector worker.

        Args:
            frame_queue (multiprocessing.Queue): Queue for incoming frames to process.
            detection_queue (multiprocessing.Queue): Queue to push processed detection results.
            detection_threshold_value (multiprocessing.Value): Shared float for live tuning.
            device_str (str): "cuda" or "cpu" string.
            ready_event (multiprocessing.Event): Event to signal when model is loaded.
        """
        super().__init__()
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.shared_threshold = detection_threshold_value
        self.device_str = device_str
        self.ready_event = ready_event
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
            self.ready_event.set()  # Signal that the model is ready
            
            while True:
                try:
                    # Expecting tuple: (camera_id, frame)
                    item = self.frame_queue.get()
                except Exception:
                    break

                # Exit signal
                if item is None:
                    print("Detector process stopping...")
                    break
                
                camera_id, frame = item

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

                # Output tuple: (camera_id, detections)
                self.detection_queue.put((camera_id, new_det))
                
        except Exception as e:
            print(f"!!!! CRITICAL ERROR in Detector Process !!!!: {e}")

