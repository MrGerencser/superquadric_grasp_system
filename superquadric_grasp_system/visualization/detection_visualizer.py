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

        # PNG saving settings
        self.save_png = True  # Enable PNG saving for testing
        self.png_counter = 0
        self.png_save_interval = 1  # Save every 1 frame

    def draw_detections(self, frame: np.ndarray, results: Any, camera_name: str = "") -> np.ndarray:
        """Draw YOLO detection boxes and labels on frame"""
        try:
            if results is None or results.boxes is None:
                return frame
            
            frame_copy = frame.copy()
            
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Scale text based on image resolution for better readability
            img_height, img_width = frame.shape[:2]
            scale_factor = max(img_width, img_height) / 1000.0  # Adaptive scaling
            font_scale = max(0.8, scale_factor * 0.8)  # Minimum 0.8, scales with resolution
            thickness = max(2, int(scale_factor * 2.5))  # Minimum 2, scales with resolution
            box_thickness = max(3, int(scale_factor * 4))  # Thicker boxes for high-res
            
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = [int(v) for v in box]
                class_name = self.class_names.get(class_id, f'Class_{class_id}')
                color = self.colors[class_id % len(self.colors)]
                
                # Clamp coordinates to image bounds (safety check)
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    print(f"Invalid box dimensions: ({x1}, {y1}) to ({x2}, {y2})")
                    continue
                
                # Draw bounding box with adaptive thickness
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, box_thickness)
                
                # Draw label with background - adaptive font size
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Background rectangle for label with padding
                padding = max(5, int(scale_factor * 8))
                
                # Ensure label background doesn't go outside image bounds
                label_y1 = max(label_size[1] + padding, y1 - label_size[1] - padding*2)
                label_x2 = min(img_width, x1 + label_size[0] + padding*2)
                
                cv2.rectangle(frame_copy, (x1, label_y1), (label_x2, y1), color, -1)
                
                # Label text with high-res font
                text_y = max(y1 - padding, label_size[1] + padding)
                cv2.putText(frame_copy, label, (x1 + padding, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            return frame_copy
            
        except Exception as e:
            print(f"Error drawing detections: {e}")
            import traceback
            traceback.print_exc()
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
                             resize_to: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Create side-by-side visualization of two camera feeds - defaults to high-res"""
        try:
            # Draw detections on both frames (at original resolution)
            display1 = self.draw_detections(frame1, results1, "Camera 1")
            display2 = self.draw_detections(frame2, results2, "Camera 2")
            
            # Add frame overlays
            display1 = self._add_frame_overlay(display1, "Camera 1")
            display2 = self._add_frame_overlay(display2, "Camera 2")
            
            # Only resize if specifically requested (default is high-res)
            if resize_to:
                display1 = cv2.resize(display1, resize_to)
                display2 = cv2.resize(display2, resize_to)
            
            # Combine horizontally
            combined = cv2.hconcat([display1, display2])

            # Save as PNG for testing
            # self._save_as_png(combined)
            
            return combined
            
        except Exception as e:
            print(f"Error creating combined frame: {e}")
            # Return error frame with appropriate size
            if resize_to:
                error_frame = np.zeros((resize_to[1], resize_to[0] * 2, 3), dtype=np.uint8)
            else:
                # Use original frame size for error frame
                h, w = frame1.shape[:2] if frame1 is not None else (480, 640)
                error_frame = np.zeros((h, w * 2, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Display Error", (error_frame.shape[1]//4, error_frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
            return error_frame
    
    def _add_frame_overlay(self, frame: np.ndarray, camera_name: str) -> np.ndarray:
        """Add timestamp and frame info overlay"""
        try:
            import time
            
            timestamp_str = time.strftime("%H:%M:%S", time.localtime())
            
            # Scale text based on image resolution
            img_height, img_width = frame.shape[:2]
            scale_factor = max(img_width, img_height) / 1000.0
            font_scale = max(0.6, scale_factor * 0.7)
            thickness = max(2, int(scale_factor * 2))
            
            # Add a semi-transparent background for better readability
            overlay_text = f"{camera_name} - {timestamp_str}"
            text_size = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Background rectangle
            cv2.rectangle(frame, (15, 15), (text_size[0] + 35, int(50 * scale_factor) + 15), 
                        (0, 0, 0), -1)  # Black background
            cv2.rectangle(frame, (15, 15), (text_size[0] + 35, int(50 * scale_factor) + 15), 
                        (0, 255, 0), 2)  # Green border
            
            # Camera name and timestamp with adaptive scaling
            cv2.putText(frame, overlay_text, 
                    (20, int(40 * scale_factor)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            
            return frame
            
        except Exception as e:
            print(f"Error adding frame overlay: {e}")
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

    def _save_as_png(self, combined_frame: np.ndarray):
        """Save current frame as high-quality PNG for testing purposes"""
        try:
            if not self.save_png:
                return

            self.png_counter += 1

            # Save every N frames to avoid too many files
            if self.png_counter % self.png_save_interval != 0:
                return
                
            import os
            import time
            
            # Create output directory
            png_dir = "/home/chris/franka_ros2_ws/src/superquadric_grasp_system/png_captures"
            os.makedirs(png_dir, exist_ok=True)

            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save as high-quality PNG instead of PDF to avoid matplotlib conflicts
            image_path = f"{png_dir}/detection_frame_{timestamp}_count_{self.png_counter}.png"
            
            # Save as lossless PNG with maximum quality
            success = cv2.imwrite(image_path, combined_frame, [
                cv2.IMWRITE_PNG_COMPRESSION, 0,  # No compression for max quality
                cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
            ])
            
            if success:
                print(f"Saved high-res PNG: {image_path}")
            else:
                print(f"Failed to save image: {image_path}")
            
        except Exception as e:
            print(f"Error saving image: {e}")
            import traceback
            traceback.print_exc()