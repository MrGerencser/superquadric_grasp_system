import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from typing import Dict, Any
import yaml
from ament_index_python.packages import get_package_share_directory
import os
import sys

sys.path.insert(0, os.path.dirname(__file__)) 

from superquadric_grasp_system.managers.camera_manager import CameraManager
from superquadric_grasp_system.managers.detection_manager import DetectionManager
from superquadric_grasp_system.managers.point_cloud_manager import PointCloudManager
from superquadric_grasp_system.managers.estimators.icp_estimator import ICPEstimator
from superquadric_grasp_system.managers.estimators.superquadric_estimator import SuperquadricEstimator
from superquadric_grasp_system.managers.config_manager import ConfigManager

class PerceptionNode(Node):
    """Unified perception node supporting multiple pose estimation and detection methods"""
    
    def __init__(self, config_file: str = None):
        super().__init__('perception_node')
        
        # Single config manager handles everything
        self.config = ConfigManager(config_file)

        self._initialize_managers()
        self._setup_processing()
        
        self.get_logger().info("Perception Node initialized successfully")
        
    # --------------------------------------- INITIALIZATION --------------------------------------------

    def _initialize_managers(self):
        """Initialize managers with ConfigManager properties"""
        # Camera manager
        self.camera_manager = CameraManager(
            node=self,
            camera1_sn=self.config.camera1_sn,
            camera2_sn=self.config.camera2_sn,
            resolution=self.config.camera_resolution,
            transform_file_path=self.config.transform_file_path,
            device=self.config.device
        )
        
        # Detection manager
        self.detection_manager = DetectionManager(
            node=self,
            model_path=self.config.yolo_model_path,
            confidence_threshold=self.config.confidence_threshold,
            classes=self.config.detection_classes,
            class_names=self.config.class_names,
            device=self.config.device,
            web_enabled=self.config.web_enabled,
            web_port=self.config.web_port,
            web_dir=self.config.web_dir
        )
        
        # Point cloud manager
        self.point_cloud_manager = PointCloudManager(
            node=self,
            device=self.config.device,
            voxel_size=self.config.voxel_size,
            workspace_bounds=self.config.workspace_bounds,
            distance_threshold=self.config.distance_threshold,
            require_both_cameras=getattr(self.config, 'require_both_cameras', True),
            publish_enabled=self.config.publish_point_clouds,
            poisson_reconstruction=self.config.poisson_reconstruction,
            outlier_removal=self.config.outlier_removal,
            voxel_downsample_size=self.config.voxel_downsample_size,
            visualize_fused_workspace=getattr(self.config, 'visualize_fused_workspace', False)
        )
        
        # Pose estimator - method-specific
        if self.config.pose_method == 'icp':
            self.pose_estimator = ICPEstimator(
                node=self,
                model_folder_path=self.config.model_folder_path,
                distance_threshold=self.config.icp_distance_threshold,
                max_iterations=self.config.icp_max_iterations,
                convergence_threshold=self.config.icp_convergence_threshold,
                class_names=self.config.class_names,
                poisson_reconstruction=self.config.poisson_reconstruction,
                outlier_removal=self.config.outlier_removal,
                voxel_downsample_size=self.config.voxel_downsample_size,
                visualize_alignment=self.config.visualize_icp_alignment,
                visualize_grasps=self.config.visualize_grasp_poses,
                max_grasp_candidates=getattr(self.config, 'max_grasp_candidates', 3)
            )
        else:
            self.pose_estimator = SuperquadricEstimator(
                node=self,
                outlier_ratio=self.config.outlier_ratio,
                use_kmeans_clustering=self.config.use_kmeans_clustering,
                gripper_jaw_length=self.config.gripper_jaw_length,
                gripper_max_opening=self.config.gripper_max_opening,
                class_names=self.config.class_names,
                poisson_reconstruction=self.config.poisson_reconstruction,
                outlier_removal=self.config.outlier_removal,
                voxel_downsample_size=self.config.voxel_downsample_size,
                enable_superquadric_fit_visualization=self.config.enable_superquadric_fit_visualization,
                enable_all_valid_grasps_visualization=self.config.enable_all_valid_grasps_visualization,
                enable_best_grasps_visualization=self.config.enable_best_grasps_visualization,
                enable_support_test_visualization=self.config.enable_support_test_visualization,
                enable_collision_test_visualization=self.config.enable_collision_test_visualization,
            )
            
        managers = [
            self.camera_manager,
            self.detection_manager, 
            self.point_cloud_manager,
            self.pose_estimator
        ]
        
        for manager in managers:
            if not manager.initialize():
                raise RuntimeError(f"Failed to initialize {manager.__class__.__name__}")
        
        self.get_logger().info("All managers initialized successfully")
    
    def _setup_processing(self):
        """Setup processing timer"""
        # Use ConfigManager properties
        self.processing_rate = self.config.processing_rate
        self.pose_estimation_method = self.config.pose_method
        
        # Threading setup
        self.frames_lock = threading.Lock()
        self.latest_frames = {'camera1': None, 'camera2': None}
        self.latest_detection_results = (None, None)
        self.latest_point_cloud_data = None
        self.live_capture_enabled = False
        self.live_capture_thread = None
        
        # Setup timer and web interface
        self.timer = self.create_timer(1.0/self.processing_rate, self.process_frames)
        
        if self.config.web_enabled:
            self.web_update_rate = self.config.web_update_rate
            self._start_live_capture()
            
    # --------------------------------------- MAIN PROCESSING --------------------------------------------
    
    def process_frames(self):
        """Main processing loop"""
        try:
            start_time = time.time()
                    
            # Step 1: Capture camera data
            frame1, frame2 = self.camera_manager.capture_frames()
            depth1, depth2 = self.camera_manager.get_depth_maps()
            
            # Step 2: Process point clouds
            pc1_cropped, pc2_cropped, fused_workspace_np, pcd_fused_workspace = self.point_cloud_manager.process_point_clouds(self.camera_manager) 

            # Step 3: Run object detection
            results1, results2, class_ids1, class_ids2 = self.detection_manager.detect_objects([frame1, frame2])

            # Step 4: Process object point clouds - pass fused_workspace_np for subtraction
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

            object_point_clouds = point_cloud_data.get('object_point_clouds', [])
            object_classes = point_cloud_data.get('object_classes', [])

            # Step 5: Process detected objects for pose estimation
            workspace_cloud = None
            if object_point_clouds:
                workspace_cloud = point_cloud_data.get('fused_workspace', None)

            success = self.pose_estimator.process_objects(
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
                self.pose_estimator,
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