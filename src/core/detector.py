import time
import multiprocessing
import numpy as np
import traceback
from ultralytics import YOLO
from utils.config import Config
from utils.logger import get_logger

logger = get_logger()

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
        logger.info(f"[{self.__class__.__name__}] Process starting on {self.device_str}...")
        
        try:
            # Initialize model within the process context
            if self.device_str == 'cpu':
                from utils.torch_utils import configure_cpu_inference
                configure_cpu_inference(Config.DETECTOR_NUM_THREADS)
            detector = YOLO(Config.YOLO_MODEL_PATH)
            
            # Warmup is handled implicitly by the first inference, 
            # but we signal ready immediately after load.
            logger.info(f"[{self.__class__.__name__}] YOLOv8 Loaded Successfully.")
            self.ready_event.set()

            # Timing statistics, logged periodically so CPU overload is visible
            stat_n, stat_busy, stat_t0 = 0, 0.0, time.time()

            while True:
                try:
                    # Blocking get
                    item = self.frame_queue.get()
                except (EOFError, BrokenPipeError):
                    break
                except Exception:
                    continue

                if item is None:
                    logger.info(f"[{self.__class__.__name__}] Stopping...")
                    break
                
                # Strict Unpacking: (cam_id, frame_id, frame)
                try:
                    cam_id, frame_id, frame = item
                except ValueError:
                    logger.warning(f"[{self.__class__.__name__}] Malformed item in queue, skipping.")
                    continue

                # Run Inference
                # Optimization: Pass 'conf' directly to the model.
                # This lets the internal NMS engine filter low-confidence boxes on the GPU (if available),
                # avoiding unnecessary memory transfer and CPU filtering.
                current_thresh = self.shared_threshold.value
                t_start = time.time()

                results = detector(
                    frame, 
                    classes=0, 
                    conf=current_thresh,
                    imgsz=Config.DETECTION_IMG_SIZE,
                    verbose=False,
                    device=self.device_str
                )

                final_dets = self.EMPTY_DETECTION

                if results and results[0].boxes is not None:
                    # Ultralytics boxes.data is (N, 6) -> [x1, y1, x2, y2, conf, cls]
                    # Since we already filtered by conf and classes=0, we just take the data.
                    det = results[0].boxes.data.cpu().numpy()

                    if det.size > 0:
                        # Ensure exactly 6 columns
                        final_dets = det[:, :6]

                # Push result: (cam_id, frame_id, detections)
                self.detection_queue.put((cam_id, frame_id, final_dets))

                stat_n += 1
                stat_busy += time.time() - t_start
                if time.time() - stat_t0 >= 10.0:
                    elapsed = time.time() - stat_t0
                    logger.info(f"[{self.__class__.__name__}] {stat_n / elapsed:.1f} frames/s, "
                                f"{stat_busy / stat_n * 1000:.0f} ms/frame, busy {stat_busy / elapsed:.0%}")
                    stat_n, stat_busy, stat_t0 = 0, 0.0, time.time()

        except Exception as e:
            logger.critical(f"CRITICAL ERROR in Detector Process: {e}")
            traceback.print_exc()
