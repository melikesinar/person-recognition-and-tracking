import multiprocessing
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Optional, Any
from utils.config import Config

class DetectorWorker(multiprocessing.Process):
    """
    Manages the YOLOv8 object detection model in a dedicated background process.
    Receives frames via a queue and pushes detection results to another queue.
    """

    def __init__(self, 
                 frame_queue: multiprocessing.Queue, 
                 detection_queue: multiprocessing.Queue, 
                 detection_threshold_value: multiprocessing.Value, 
                 device_str: str, 
                 ready_event: multiprocessing.Event):
        """
        Args:
            frame_queue: Input queue receiving (cam_id, frame_id, frame).
            detection_queue: Output queue sending (cam_id, frame_id, detections).
            detection_threshold_value: Shared float for dynamic threshold adjustment.
            device_str: 'cuda' or 'cpu'.
            ready_event: Signal event set when model is fully loaded.
        """
        super().__init__()
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.shared_threshold = detection_threshold_value
        self.device_str = device_str
        self.ready_event = ready_event
        self.daemon = True
        
        # Pre-allocate empty result for valid frames with no detections
        self.EMPTY_DETECTION = np.zeros((0, 6), dtype=np.float32)

    def stop(self):
        """Signals the worker to exit and waits for it to join."""
        self.frame_queue.put(None)
        self.join()

    def run(self):
        print(f"[{self.__class__.__name__}] Process starting on {self.device_str}...")
        
        try:
            # Initialize model within the process context
            detector = YOLO(Config.YOLO_MODEL_PATH)
            
            # Warmup or simple check could go here if needed
            print(f"[{self.__class__.__name__}] YOLOv8 Loaded Successfully.")
            self.ready_event.set()
            
            while True:
                try:
                    # Blocking get
                    item = self.frame_queue.get()
                except (EOFError, BrokenPipeError):
                    break
                except Exception:
                    continue

                if item is None:
                    print(f"[{self.__class__.__name__}] Stopping...")
                    break
                
                # Strict Unpacking: (cam_id, frame_id, frame)
                try:
                    cam_id, frame_id, frame = item
                except ValueError:
                    # Malformed item, skip
                    continue

                # Run Inference
                # verbose=False reduces log spam
                # stream=False is default, returns list of Results
                results = detector(frame, classes=0, verbose=False, device=self.device_str)

                final_dets = self.EMPTY_DETECTION

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    # Ultralytics boxes.data is (N, 6) -> [x1, y1, x2, y2, conf, cls]
                    det = boxes.data.cpu().numpy()

                    if det.size > 0:
                        # Apply dynamic threshold filtering
                        thresh = self.shared_threshold.value
                        mask = det[:, 4] >= thresh
                        filtered_det = det[mask]
                        
                        if filtered_det.size > 0:
                            # Ensure exactly 6 columns (sometimes frameworks add extra info)
                            final_dets = filtered_det[:, :6]

                # Push result: (cam_id, frame_id, detections)
                self.detection_queue.put((cam_id, frame_id, final_dets))

        except Exception as e:
            print(f"!!!! CRITICAL ERROR in Detector Process !!!!: {e}")
            import traceback
            traceback.print_exc()