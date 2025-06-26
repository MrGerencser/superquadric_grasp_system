import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2
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

        # Load configuration
        self.config = self._load_config(config_file)
        
        # Extract main settings
        self.processing_rate = self.config['perception']['processing_rate']
        self.visualization_rate = self.config['perception']['visualization_rate']
        
        # Extract web interface settings
        web_config = self.config['perception'].get('web_interface', {})
        self.enable_web_visualization = web_config.get('enabled', False)
        self.web_port = web_config.get('port', 8080)
        self.web_dir = web_config.get('web_dir', '/tmp/grasp_system_live')
        self.web_update_rate = web_config.get('update_rate', 15.0)
        self.web_image_quality = web_config.get('image_quality', 85)
        
        # Extract pose estimation method for logging
        self.pose_estimation_method = self.config['perception']['pose_estimation']['method']
        
        # Extract visualization settings from pose_estimation config
        pose_config = self.config['perception']['pose_estimation']
        self.enable_workspace_visualization = False  # Default - can be added to config later
        self.enable_detected_object_clouds_visualization = pose_config.get('enable_detected_object_clouds_visualization', False)
        
        # Superquadric-specific visualization
        superquadric_config = pose_config.get('superquadric', {})
        self.enable_detected_object_clouds_visualization = superquadric_config.get('enable_detected_object_clouds_visualization', False)
        self.enable_superquadric_fit_visualization = superquadric_config.get('enable_superquadric_fit_visualization', False)
        self.enable_support_test_visualization = superquadric_config.get('enable_support_test_visualization', False)
        self.enable_collision_test_visualization = superquadric_config.get('enable_collision_test_visualization', False)
        self.enable_all_valid_grasps_visualization = superquadric_config.get('enable_all_valid_grasps_visualization', False)
        self.enable_best_grasps_visualization = superquadric_config.get('enable_best_grasps_visualization', False)
        
        # ICP-specific visualization
        icp_config = pose_config.get('icp', {})
        self.visualize_icp_alignment = icp_config.get('visualize_icp_alignment', False)

        # Log configuration
        self._log_configuration()
        
        # Initialize manager configs
        self._initialize_manager_configs()
        
        # Initialize managers
        self._initialize_managers()
        
        self.get_logger().info("Perception Node initialized successfully")

    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_file is None:
            # Use the config file in the same directory
            config_file = os.path.join(os.path.dirname(__file__), 'perception_config.yaml')

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
            # Fall back to default configuration
            return self._get_default_config()

    def _determine_device(self, device_str: str) -> str:
        """Determine the device to use based on the input string"""
        if device_str == 'cuda':
            return 'cuda'
        elif device_str == 'cpu':
            return 'cpu'
        else:
            # Default to auto-detecting CUDA if available
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
                else:
                    return 'cpu'
            except ImportError:
                return 'cpu'

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if config file loading fails"""
        return {
            'perception': {
                'processing_rate': 10.0,
                'visualization_rate': 15.0,
                'device': 'cpu',
                'web_interface': {
                    'enabled': False,
                    'port': 8080,
                    'web_dir': '/tmp/grasp_system_live',
                    'update_rate': 15.0,
                    'image_quality': 85
                },
                'camera': {
                    'camera1_sn': 33137761,
                    'camera2_sn': 36829049,
                    'resolution': 'HD2K',
                    'device': 'cpu'
                },
                'detection': {
                    'method': 'yolo',
                    'confidence_threshold': 0.1,
                    'device': 'cpu'
                },
                'point_cloud': {
                    'voxel_size': 0.003,
                    'distance_threshold': 0.4,
                    'device': 'cpu'
                },
                'pose_estimation': {
                    'method': 'superquadric',
                    'superquadric_enabled': True,
                    'grasp_planning_enabled': True,
                    'device': 'cpu'
                }
            }
        }

    def _log_configuration(self):
        """Log the current configuration"""
        config = self.config['perception']
        
        self.get_logger().info(f"Initializing perception node with:")
        self.get_logger().info(f"  - Pose estimation: {self.pose_estimation_method}")
        self.get_logger().info(f"  - Detection method: {config['detection']['method']}")
        
        # Handle pose estimation method-specific logging
        if self.pose_estimation_method == 'superquadric':
            self.get_logger().info(f"  - Grasp planning: {'enabled' if config['pose_estimation'].get('grasp_planning_enabled', False) else 'disabled'}")
            self.get_logger().info(f"  - K-means clustering: {'enabled' if config['pose_estimation'].get('use_kmeans_clustering', False) else 'disabled'}")
        elif self.pose_estimation_method == 'icp':
            self.get_logger().info(f"  - ICP visualization alignment: {'enabled' if self.visualize_icp_alignment else 'disabled'}")

        self.get_logger().info(f"  - Main processing rate: {config['processing_rate']} Hz")
        self.get_logger().info(f"  - Visualization rate: {config['visualization_rate']} Hz")
        
        # Web interface logging
        self.get_logger().info(f"Web interface settings:")
        self.get_logger().info(f"  - Web interface: {'enabled' if self.enable_web_visualization else 'disabled'}")
        if self.enable_web_visualization:
            self.get_logger().info(f"  - Web port: {self.web_port}")
            self.get_logger().info(f"  - Web directory: {self.web_dir}")
            self.get_logger().info(f"  - Update rate: {self.web_update_rate} FPS")
            self.get_logger().info(f"  - Image quality: {self.web_image_quality}%")
        
        # Visualization logging for superquadric method
        if self.pose_estimation_method == 'superquadric':
            self.get_logger().info(f"Superquadric visualization settings:")
            self.get_logger().info(f"  - Detected object clouds: {'enabled' if self.enable_detected_object_clouds_visualization else 'disabled'}")
            self.get_logger().info(f"  - Superquadric fit: {'enabled' if self.enable_superquadric_fit_visualization else 'disabled'}")
            self.get_logger().info(f"  - Support test: {'enabled' if self.enable_support_test_visualization else 'disabled'}")
            self.get_logger().info(f"  - Collision test: {'enabled' if self.enable_collision_test_visualization else 'disabled'}")
            self.get_logger().info(f"  - All valid grasps: {'enabled' if self.enable_all_valid_grasps_visualization else 'disabled'}")
            self.get_logger().info(f"  - Best grasps: {'enabled' if self.enable_best_grasps_visualization else 'disabled'}")

    def _initialize_manager_configs(self):
        """Initialize manager-specific configurations"""
        config = self.config['perception']
        
        # Detection manager config
        self.detection_config = config['detection'].copy()
        self.detection_config.update({
            'enable_web_visualization': self.enable_web_visualization,
            'enable_detection_visualization': self.enable_web_visualization,
            'web_port': self.web_port,
            'web_dir': self.web_dir,
            'web_update_rate': self.web_update_rate,
            'web_image_quality': self.web_image_quality
        })
        
        # Pose estimation manager config
        self.pose_config = config['pose_estimation'].copy()
        self.pose_config.update({
            # Superquadric visualization flags
            'enable_all_valid_grasps_visualization': self.enable_all_valid_grasps_visualization,
            'enable_best_grasps_visualization': self.enable_best_grasps_visualization,
            'enable_support_test_visualization': self.enable_support_test_visualization,
            'enable_collision_test_visualization': self.enable_collision_test_visualization,
            'enable_superquadric_fit_visualization': self.enable_superquadric_fit_visualization,
            'enable_detected_object_clouds_visualization': self.enable_detected_object_clouds_visualization,
            # ICP visualization flags
            'visualize_icp_alignment': self.visualize_icp_alignment,
            # Web interface config for pose estimation
            'web_enabled': self.enable_web_visualization,
            'web_port': self.web_port,
            'web_dir': self.web_dir
        })
    
        # Point cloud manager config
        self.point_cloud_config = config['point_cloud'].copy()
        self.point_cloud_config.update({
            'visualize_fused_workspace': self.enable_workspace_visualization
        })
        
        # Camera manager config
        self.camera_config = config['camera'].copy()
        
        # Log the resolution being used
        resolution = self.camera_config.get('resolution', 'HD2K')
        self.get_logger().info(f"Camera resolution configured: {resolution}")

    def _initialize_managers(self):
        """Initialize all managers"""
        self.camera_manager = CameraManager(self, self.camera_config)
        self.detection_manager = DetectionManager(self, self.detection_config)
        self.point_cloud_manager = PointCloudManager(self, self.point_cloud_config)
        self.pose_estimation_manager = PoseEstimationManager(self, self.pose_config)

        # Shared state for visualization thread
        self.latest_frames = {'camera1': None, 'camera2': None}
        self.latest_detection_results = (None, None)
        self.latest_point_cloud_data = None
        self.frames_lock = threading.Lock()
        
        # High-frequency visualization control
        self.live_capture_enabled = False
        self.live_capture_thread = None

        # Initialize all managers
        if not self.initialize_managers():
            self.get_logger().error("Failed to initialize managers")
            raise RuntimeError("Failed to initialize managers")

        # Set up publishers and subscribers
        self.setup_publishers_and_subscribers()

        # Start main processing loop
        self.timer = self.create_timer(1.0/self.processing_rate, self.process_frames)
        
        # Start high-frequency visualization if web interface is enabled
        if self.enable_web_visualization:
            self._start_live_capture()
        
        self.get_logger().info("Perception Node initialized successfully")
        
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
    
    def initialize_managers(self) -> bool:
        """Initialize all managers"""
        managers = [
            ('camera', self.camera_manager),
            ('detection', self.detection_manager),
            ('point_cloud', self.point_cloud_manager),
            ('pose_estimation', self.pose_estimation_manager)
        ]
        
        for name, manager in managers:
            try:
                if not manager.initialize():
                    self.get_logger().error(f"Failed to initialize {name} manager")
                    return False
                self.get_logger().info(f"{name.title()} manager initialized")
            except Exception as e:
                self.get_logger().error(f"Error initializing {name} manager: {e}")
                return False

        return True

    def setup_publishers_and_subscribers(self):
        """Set up ROS publishers and subscribers (without point cloud publishers)"""
        # Publisher for poses
        self.pose_publisher = self.create_publisher(
            PoseStamped, '/perception/object_pose', 1)
        
        # Subscriber for grasp execution state
        self.execution_state_subscriber = self.create_subscription(
            Bool, '/robot/grasp_executing',
            self.grasp_execution_callback, 1
        )
        
        self.get_logger().info("Publishers and subscribers set up")

    def grasp_execution_callback(self, msg):
        """Callback for grasp execution state"""
        # Can pause processing during grasp execution if needed
        pass

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