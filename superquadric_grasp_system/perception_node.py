import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, Bool
import sensor_msgs_py.point_cloud2 as pc2
from typing import Dict, Any, Optional
import yaml
from ament_index_python.packages import get_package_share_directory
import os
import sys

sys.path.insert(0, os.path.dirname(__file__)) 

from superquadric_grasp_system.managers.camera_manager import CameraManager
from superquadric_grasp_system.managers.detection_manager import DetectionManager
from superquadric_grasp_system.managers.point_cloud_manager import PointCloudManager
from superquadric_grasp_system.managers.pose_estimation_manager import PoseEstimationManager

class PerceptionNode(Node):
    """Unified perception node supporting multiple pose estimation and detection methods"""
    
    def __init__(self, config_file: str = None):
        super().__init__('perception_node')

        # Load and process configuration
        self.config = self._load_config(config_file)
        base_config = self.config['perception']
        
        # Extract core settings
        self.processing_rate = base_config['processing_rate']
        self.visualization_rate = base_config['visualization_rate']
        self.pose_estimation_method = base_config['pose_estimation']['method']
        
        # Extract all visualization settings
        self.viz_config = self._extract_visualization_config()
        
        # Log configuration and initialize
        self._log_configuration()
        self._initialize_managers()
        
        self.get_logger().info("Perception Node initialized successfully")

    # --------------------------------------- MAIN PROCESSING --------------------------------------------
    
    def process_frames(self):
        """Main processing loop (runs at 10 Hz for heavy processing)"""
        try:
            start_time = time.time()
                    
            # Step 1: Capture frames from cameras
            frame1, frame2 = self.camera_manager.capture_frames()
            if frame1 is None or frame2 is None:
                return

            # Step 2: Get depth maps
            depth1, depth2 = self.camera_manager.get_depth_maps()
            if depth1 is None or depth2 is None:
                return
            
            # Step 3: Process point clouds
            pc1_cropped, pc2_cropped, fused_workspace_np, pcd_fused_workspace = self.point_cloud_manager.process_point_clouds(self.camera_manager) 

            # Step 4: Run object detection
            frames_list = [frame1, frame2]
            results1, results2, class_ids1, class_ids2 = self.detection_manager.detect_objects(frames_list)
            if not results1 or not results2:
                return

            # Step 5: Process object point clouds - pass fused_workspace_np for subtraction
            fused_object_points, fused_object_classes, fused_objects_np, subtracted_cloud = \
                self.point_cloud_manager.extract_object_point_clouds(
                    results1, results2, class_ids1, class_ids2,
                    depth1, depth2, self.camera_manager, fused_workspace_np
                )

            # Create point_cloud_data dictionary for compatibility
            point_cloud_data = {
                'fused_workspace': fused_workspace_np,
                'fused_objects': fused_objects_np,
                'subtracted_cloud': subtracted_cloud,
                'object_point_clouds': fused_object_points,
                'object_classes': fused_object_classes
            }
            
            # Update shared state for visualization thread
            with self.frames_lock:
                self.latest_frames = {'camera1': frame1, 'camera2': frame2}
                self.latest_detection_results = (results1, results2)
                self.latest_point_cloud_data = point_cloud_data
            
            if not point_cloud_data:
                return

            object_point_clouds = point_cloud_data.get('object_point_clouds', [])
            object_classes = point_cloud_data.get('object_classes', [])

            # Step 6: Process detected objects for pose estimation
            if object_point_clouds and self.pose_estimation_manager.is_ready():
                workspace_cloud = point_cloud_data.get('fused_workspace', None)
                
                # Debug: Log data types
                self.get_logger().debug(f"Processing {len(object_point_clouds)} objects for pose estimation")
                for i, (pc, class_id) in enumerate(zip(object_point_clouds, object_classes)):
                    self.get_logger().debug(f"Object {i}: type={type(pc)}, class={class_id}")
                    if isinstance(pc, np.ndarray):
                        self.get_logger().debug(f"  - Shape: {pc.shape}")
                    elif hasattr(pc, 'points'):
                        self.get_logger().debug(f"  - Points: {len(pc.points)}")
                
                success = self.pose_estimation_manager.process_objects(
                    object_point_clouds, object_classes, workspace_cloud
                )
                
                if success:
                    self.get_logger().debug(f"Pose estimation completed successfully using {self.pose_estimation_method} method")

            # Performance monitoring
            total_time = time.time() - start_time
            if total_time > 1.0 / (self.processing_rate * 0.8):  # Log if running slow
                fps = 1.0 / total_time if total_time > 0 else 0
                self.get_logger().debug(f"Main processing time: {total_time:.3f}s (FPS: {fps:.1f})")

        except Exception as e:
            self.get_logger().error(f"Error in process_frames: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    # --------------------------------------- CONFIGURATION --------------------------------------------
    
    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_file is None:
            # Use ROS2 package share directory to find config file
            try:
                package_share_directory = get_package_share_directory('superquadric_grasp_system')
                config_file = os.path.join(package_share_directory, 'config', 'perception_config.yaml')
            except Exception:
                # Fallback to local directory if package not found (development mode)
                config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'perception_config.yaml')

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Determine device
            device_str = config['perception']['device']
            config['perception']['device'] = self._determine_device(device_str)
            
            # Add device to all manager configs
            for manager_config in ['camera', 'detection', 'point_cloud', 'pose_estimation']:
                config['perception'][manager_config]['device'] = config['perception']['device']
            
            self.get_logger().info(f"Loaded configuration from {config_file}")
            return config
            
        except Exception as e:
            self.get_logger().error(f"Failed to load config from {config_file}: {e}")
            raise RuntimeError(f"Could not load configuration file: {config_file}")
        
    def _extract_visualization_config(self) -> Dict[str, Any]:
        """Extract and consolidate all visualization settings"""
        pose_config = self.config['perception']['pose_estimation']
        web_config = self.config['perception'].get('web_interface', {})
        
        viz_config = {
            # Web interface
            'web_enabled': web_config.get('enabled', False),
            'web_port': web_config.get('port', 8080),
            'web_dir': web_config.get('web_dir', '/tmp/grasp_system_live'),
            'web_update_rate': web_config.get('update_rate', 15.0),
            'web_image_quality': web_config.get('image_quality', 85),
            
            # Superquadric visualizations
            **pose_config.get('superquadric', {}),
            
            # ICP visualizations  
            **pose_config.get('icp', {}),
            
            # Shared visualizations
            'enable_workspace_visualization': False,
            'enable_detected_object_clouds_visualization': pose_config.get('enable_detected_object_clouds_visualization', False)
        }
        
        return viz_config
    
    def _determine_device(self, device_str: str) -> str:
        """Determine the device to use based on the input string"""
        if device_str in ['cuda', 'cpu']:
            return device_str
        
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'
        
    def _log_configuration(self):
        """Log essential configuration information"""
        config = self.config['perception']
        viz = self.viz_config
        
        self.get_logger().info(f"Perception Node Configuration:")
        self.get_logger().info(f"  Method: {self.pose_estimation_method} | Detection: {config['detection']['method']}")
        self.get_logger().info(f"  Rates: Processing={self.processing_rate}Hz, Viz={self.visualization_rate}Hz")
        
        if viz['web_enabled']:
            self.get_logger().info(f"  Web: Enabled on port {viz['web_port']} @ {viz['web_update_rate']}FPS")
        
        # Log method-specific settings
        if self.pose_estimation_method == 'superquadric' and any(v for k, v in viz.items() if 'visualization' in k):
            enabled_viz = [k.replace('enable_', '').replace('_visualization', '') for k, v in viz.items() 
                        if 'visualization' in k and v]
            if enabled_viz:
                self.get_logger().info(f"  Superquadric visualizations: {', '.join(enabled_viz)}")
          
    # --------------------------------------- INITIALIZATION --------------------------------------------

    def _initialize_managers(self):
        """Initialize all managers with their configurations"""
        base_config = self.config['perception']
        viz_config = self.viz_config
        
        # Create managers with merged configs
        self.camera_manager = CameraManager(self, base_config['camera'])
        
        self.detection_manager = DetectionManager(self, {
            **base_config['detection'],
            **{k: v for k, v in viz_config.items() if k.startswith('web_') or k == 'enable_detection_visualization'}
        })
        
        self.point_cloud_manager = PointCloudManager(self, {
            **base_config['point_cloud'],
            'visualize_fused_workspace': viz_config.get('enable_workspace_visualization', False)
        })
        
        self.pose_estimation_manager = PoseEstimationManager(self, {
            **base_config['pose_estimation'],
            **viz_config
        })
        
        # Initialize shared state
        self._init_shared_state()
        
        # Initialize managers and setup
        if not self._init_all_managers():
            raise RuntimeError("Failed to initialize managers")
        
        self.setup_publishers_and_subscribers()
        self._setup_timers_and_threads()
            
    def _init_shared_state(self):
        """Initialize shared state for threading"""
        self.latest_frames = {'camera1': None, 'camera2': None}
        self.latest_detection_results = (None, None)
        self.latest_point_cloud_data = None
        self.frames_lock = threading.Lock()
        self.live_capture_enabled = False
        self.live_capture_thread = None

    def _init_all_managers(self) -> bool:
        """Initialize all managers in sequence"""
        managers = [
            ('camera', self.camera_manager),
            ('detection', self.detection_manager),
            ('point_cloud', self.point_cloud_manager),
            ('pose_estimation', self.pose_estimation_manager)
        ]
        
        for name, manager in managers:
            if not manager.initialize():
                self.get_logger().error(f"Failed to initialize {name} manager")
                return False
            self.get_logger().info(f"{name.title()} manager initialized")
        
        return True

    def _setup_timers_and_threads(self):
        """Setup processing timer and visualization threads"""
        self.timer = self.create_timer(1.0/self.processing_rate, self.process_frames)
        
        if self.viz_config['web_enabled']:
            self._start_live_capture()

    # --------------------------------------- ROS INTERFACE --------------------------------------------

    def setup_publishers_and_subscribers(self):
        """Set up ROS publishers and subscribers (without point cloud publishers)"""
        # Publisher for poses
        self.pose_publisher = self.create_publisher(
            PoseStamped, '/perception/object_pose', 1)
        
        # OPTIONAL: Subscriber for grasp execution state
        self.execution_state_subscriber = self.create_subscription(
            Bool, '/robot/grasp_executing',
            self.grasp_execution_callback, 1
        )
        
        self.get_logger().info("Publishers and subscribers set up")

    def grasp_execution_callback(self, msg):
        """Callback for grasp execution state"""
        # Can pause processing during grasp execution if needed
        pass
    
    # --------------------------------------- VISUALIZATION --------------------------------------------

    def _start_live_capture(self):
        """Start live capture thread for web visualization"""
        try:
            self.live_capture_enabled = True
            self.live_capture_thread = threading.Thread(target=self._live_capture_worker)
            self.live_capture_thread.daemon = True
            self.live_capture_thread.start()
            self.get_logger().info("Live capture thread started for web visualization")
        except Exception as e:
            self.get_logger().error(f"Failed to start live capture: {e}")
            self.live_capture_enabled = False    
            
    def _live_capture_worker(self):
        """Worker thread for live capture"""
        while self.live_capture_enabled:
            try:
                # Update web visualization at higher frequency
                with self.frames_lock:
                    if (self.latest_frames['camera1'] is not None and 
                        self.latest_detection_results[0] is not None):
                        self.detection_manager.update_web_visualization(
                            self.latest_frames, self.latest_detection_results
                        )
                
                time.sleep(1.0 / self.web_update_rate)  # Use web-specific update rate
            except Exception as e:
                self.get_logger().error(f"Error in live capture worker: {e}")
    
    # --------------------------------------- CLEANUP --------------------------------------------
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Stop live capture thread
            self.live_capture_enabled = False
            if self.live_capture_thread and self.live_capture_thread.is_alive():
                self.live_capture_thread.join(timeout=2.0)
            
            managers = [
                self.camera_manager,
                self.detection_manager, 
                self.point_cloud_manager,
                self.pose_estimation_manager
            ]
            
            for manager in managers:
                if hasattr(manager, 'cleanup'):
                    manager.cleanup()
                    
            self.get_logger().info("Perception Node cleaned up")
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")

    def destroy_node(self):
        """Clean shutdown"""
        self.cleanup()
        super().destroy_node()
        

def main(args=None):
    rclpy.init(args=args)
    
    node = None
    try:
        node = PerceptionNode()
        
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
    
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("Shutting down due to keyboard interrupt")
    except Exception as e:
        if node:
            node.get_logger().error(f"Error in main: {e}")
        else:
            print(f"Error in main (node not initialized): {e}")
    
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()