import numpy as np
import cv2
from typing import Any, Dict, List, Tuple, Optional

class DetectionVisualizer:
    """Handles visualization of YOLO detection results"""
    
    def __init__(self, class_names: Dict[int, str] = None):
        # Default class names from your system
        self.class_names = class_names or {
            0: 'Cone', 1: 'Cup', 2: 'Mallet', 3: 'Screw Driver', 4: 'Sunscreen'
        }
        
        # Colors for different classes (BGR format for OpenCV)
        self.colors = [
            (0, 255, 0),    # Green - Cone
            (255, 0, 0),    # Blue - Cup  
            (0, 0, 255),    # Red - Mallet
            (255, 255, 0),  # Cyan - Screw Driver
            (255, 0, 255)   # Magenta - Sunscreen
        ]
    
    def draw_detections(self, frame: np.ndarray, results: Any, camera_name: str = "") -> np.ndarray:
        """Draw YOLO detection boxes and labels on frame"""
        try:
            if results is None or results.boxes is None:
                return frame
            
            frame_copy = frame.copy()
            
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = [int(v) for v in box]
                class_name = self.class_names.get(class_id, f'Class_{class_id}')
                color = self.colors[class_id % len(self.colors)]
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
                
                # Draw label with background
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Background rectangle for label
                cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0] + 10, y1), color, -1)
                
                # Label text
                cv2.putText(frame_copy, label, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add camera name if provided
            if camera_name:
                cv2.putText(frame_copy, camera_name, (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            return frame_copy
            
        except Exception as e:
            print(f"Error drawing detections: {e}")
            return frame
    
    def draw_segmentation_masks(self, frame: np.ndarray, results: Any, alpha: float = 0.5) -> np.ndarray:
        """Draw segmentation masks if available"""
        try:
            if results is None or not hasattr(results, 'masks') or results.masks is None:
                return frame
            
            frame_copy = frame.copy()
            masks = results.masks.data.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for mask, class_id in zip(masks, class_ids):
                color = self.colors[class_id % len(self.colors)]
                
                # Resize mask to frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_bool = mask_resized > 0.5
                
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask_bool] = color
                
                # Blend with original frame
                frame_copy = cv2.addWeighted(frame_copy, 1.0, colored_mask, alpha, 0)
            
            return frame_copy
            
        except Exception as e:
            print(f"Error drawing segmentation masks: {e}")
            return frame
    
    def create_combined_frame(self, frame1: np.ndarray, frame2: np.ndarray, 
                             results1: Any, results2: Any,
                             resize_to: Tuple[int, int] = (640, 360)) -> np.ndarray:
        """Create side-by-side visualization of two camera feeds"""
        try:
            # Draw detections on both frames
            display1 = self.draw_detections(frame1, results1, "Camera 1")
            display2 = self.draw_detections(frame2, results2, "Camera 2")
            
            # Add frame overlays
            display1 = self._add_frame_overlay(display1, "Camera 1")
            display2 = self._add_frame_overlay(display2, "Camera 2")
            
            # Resize frames
            if resize_to:
                display1 = cv2.resize(display1, resize_to)
                display2 = cv2.resize(display2, resize_to)
            
            # Combine horizontally
            combined = cv2.hconcat([display1, display2])
            
            return combined
            
        except Exception as e:
            print(f"Error creating combined frame: {e}")
            # Return error frame
            error_frame = np.zeros((400, 1280, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Display Error", (400, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
            return error_frame
    
    def _add_frame_overlay(self, frame: np.ndarray, camera_name: str) -> np.ndarray:
        """Add timestamp and frame info overlay"""
        try:
            import time
            
            timestamp_str = time.strftime("%H:%M:%S", time.localtime())
            
            # Camera name and timestamp
            cv2.putText(frame, f"{camera_name} - {timestamp_str}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error adding frame overlay: {e}")
            return frame
    
    def add_title_bar(self, frame: np.ndarray, title: str = "Superquadric Grasp System", 
                     frame_count: int = 0) -> np.ndarray:
        """Add title bar to the top of the frame"""
        try:
            border_height = 40
            total_height = frame.shape[0] + border_height
            total_width = frame.shape[1]
            
            final_display = np.zeros((total_height, total_width, 3), dtype=np.uint8)
            final_display[border_height:, :] = frame
            
            # Green title bar
            cv2.rectangle(final_display, (0, 0), (total_width, border_height), (0, 100, 0), -1)
            cv2.putText(final_display, f"{title} - Frame {frame_count}", 
                       (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            return final_display
            
        except Exception as e:
            print(f"Error adding title bar: {e}")
            return frame
    
    def get_detection_summary(self, results: Any) -> Dict[str, int]:
        """Get summary of detections for logging/display"""
        try:
            if results is None or results.boxes is None:
                return {}
            
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            summary = {}
            
            for class_id in class_ids:
                class_name = self.class_names.get(class_id, f'Class_{class_id}')
                summary[class_name] = summary.get(class_name, 0) + 1
            
            return summary
            
        except Exception as e:
            print(f"Error getting detection summary: {e}")
            return {}
    
    def update_class_names(self, new_class_names: Dict[int, str]):
        """Update class names mapping"""
        self.class_names.update(new_class_names)
    
    def update_colors(self, new_colors: List[Tuple[int, int, int]]):
        """Update color scheme"""
        self.colors = new_colors