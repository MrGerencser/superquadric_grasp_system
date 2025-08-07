import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional, Any, Dict

class DetectionManager:
    """Manages YOLO detection and segmentation"""
    def __init__(self, node, model_path: str, confidence_threshold: float,
                 classes: List[int], class_names: Dict[int, str],
                 device: str = "cpu", web_enabled: bool = False, 
                 web_port: int = 8080, web_dir: str = "/tmp/grasp_system_live"):
        self.node = node
        self.logger = node.get_logger()
        
        # Direct property assignment
        self.device = device
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.classes = classes
        self.class_names = class_names
        
        # Initialize as not ready
        self.model = None
        self.is_initialized = False

        # Web visualization
        self.enable_web_visualization = web_enabled
        self.web_port = web_port
        self.web_dir = web_dir
        
        # Web visualization components
        self.web_interface = None
        self.detection_visualizer = None
        
        if self.enable_web_visualization:
            self._setup_web_visualization()

    def _setup_web_visualization(self):
        """Setup web visualization components"""
        try:
            from ..visualization.web_interface import WebInterface
            from ..visualization.detection_visualizer import DetectionVisualizer
            
            self.detection_visualizer = DetectionVisualizer(self.class_names)

            # Use correct config structure
            web_interface_config = {
                'web_dir': self.web_dir,
                'port': self.web_port
            }
            self.web_interface = WebInterface(**web_interface_config)
            
            self.logger.info("Web visualization components initialized")
            
        except ImportError as e:
            self.logger.warning(f"Could not import web visualization: {e}")
            self.enable_web_visualization = False
        except Exception as e:
            self.logger.error(f"Error setting up web visualization: {e}")
            self.enable_web_visualization = False

    def initialize(self) -> bool:
        """Initialize YOLO model and web interface"""
        try:
            # Initialize YOLO model
            self.logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path).to(self.device)
            
            # Initialize web interface if enabled
            if self.enable_web_visualization and self.web_interface:
                if self.web_interface.initialize():
                    self.logger.info("Web interface initialized for detection visualization")
                else:
                    self.logger.warning("Failed to initialize web interface")
                    self.enable_web_visualization = False
            
            self.is_initialized = True
            self.logger.info(f"YOLO model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def detect_objects(self, frames: List[np.ndarray]) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
        """Run YOLO detection on frame batch"""
        try:
            if not self.is_ready():
                return None, None, np.array([]), np.array([])
            
            results_batch = self.model.track(
                source=frames,
                classes=self.classes,
                persist=True,
                retina_masks=True,
                conf=self.confidence_threshold,
                device=self.device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )
            
            results1, results2 = results_batch[0], results_batch[1]
            
            # Extract class IDs
            class_ids1 = (results1.boxes.cls.cpu().numpy() 
                         if results1.boxes is not None else np.array([]))
            class_ids2 = (results2.boxes.cls.cpu().numpy() 
                         if results2.boxes is not None else np.array([]))
            
            return results1, results2, class_ids1, class_ids2
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
            return None, None, np.array([]), np.array([])
    
    def predict_fast(self, frames: List[np.ndarray]) -> Tuple[Any, Any]:
        """Fast prediction for live visualization"""
        try:
            if not self.is_ready():
                return None, None
            
            results_batch = self.model.predict(
                source=frames,
                classes=self.classes,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
                imgsz=320  # Smaller image size for faster inference
            )
            
            return results_batch[0], results_batch[1]
            
        except Exception as e:
            self.logger.error(f"Error in fast prediction: {e}")
            return None, None
    
    def is_ready(self) -> bool:
        """Check if detection model is ready"""
        return self.is_initialized and self.model is not None
    
    def cleanup(self):
        """Clean up detection resources"""
        try:
            if self.model:
                del self.model
                torch.cuda.empty_cache()
            self.logger.info("Detection manager cleaned up")
        except Exception as e:
            self.logger.error(f"Error during detection cleanup: {e}")
    
    def update_web_visualization(self, frames_dict, detection_results):
        """Update web visualization with detection results"""
        try:
            if not self.enable_web_visualization or not self.web_interface or not self.detection_visualizer:
                self.logger.warning("Web visualization is not enabled or components are missing")
                return
                
            frame1 = frames_dict.get('camera1')
            frame2 = frames_dict.get('camera2')
            
            if frame1 is None or frame2 is None:
                return
                
            # Extract detection results
            if isinstance(detection_results, tuple) and len(detection_results) >= 2:
                results1, results2 = detection_results[0], detection_results[1]
            else:
                results1, results2 = detection_results, None
            
            # Create combined visualization
            combined_frame = self.detection_visualizer.create_combined_frame(
                frame1, frame2, results1, results2, (640, 360)
            )
            
            # Update web interface
            self.web_interface.update_frame(combined_frame, quality=85)
            
        except Exception as e:
            self.logger.error(f"Error updating web visualization: {e}")