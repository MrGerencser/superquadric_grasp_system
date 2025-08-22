import yaml
import os
from typing import Dict, Any, Optional
from ament_index_python.packages import get_package_share_directory
from rclpy.logging import get_logger

class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_file: str = None):
        self.logger = get_logger('ConfigManager')
        self._config = self._load_config(config_file)
        self._flatten_config()
    
    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_file is None:
            try:
                package_share_directory = get_package_share_directory('superquadric_grasp_system')
                config_file = os.path.join(package_share_directory, 'config', 'perception_config.yaml')
            except Exception:
                config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'perception_config.yaml')
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
        
    def _extract_yolo_classes(self, model_path: str) -> tuple[list, dict]:
        """Extract class information from YOLO model"""
        try:
            from ultralytics import YOLO
            
            # Load the model
            model = YOLO(model_path)
            
            # Get class names from the model
            class_names = model.names  # This is a dict like {0: 'class1', 1: 'class2', ...}
            
            # Create classes list (indices)
            classes = list(class_names.keys())
            
            self.logger.info(f"Extracted {len(classes)} classes from YOLO model:")
            for idx, name in class_names.items():
                self.logger.info(f"  {idx}: {name}")
            
            return classes, class_names
            
        except ImportError as e:
            self.logger.warning(f"Could not import YOLO: {e}. Using manual class configuration.")
            return None, None
        except Exception as e:
            self.logger.warning(f"Could not extract classes from YOLO model: {e}. Using manual class configuration.")
            return None, None
    
    def _flatten_config(self):
        """Flatten nested config for easy access"""
        base = self._config['perception']
        
        # Core settings
        self.processing_rate = base['processing_rate']
        self.visualization_rate = base['visualization_rate']
        self.device = self._determine_device(base['device'])
        
        # Detection settings
        detection = base['detection']
        self.detection_method = detection['method']
        self.yolo_model_path = detection['yolo_model_path']
        self.confidence_threshold = detection['confidence_threshold']
        self.detection_classes, self.class_names = self._extract_yolo_classes(self.yolo_model_path)
        if self.detection_classes is None:
            # Fallback to manual class configuration if YOLO extraction fails
            self.logger.warning("Using manual class configuration due to YOLO extraction failure.")

        
        # Camera settings
        camera = base['camera']
        self.camera1_sn = camera['camera1_sn']
        self.camera2_sn = camera['camera2_sn']
        self.camera_resolution = camera['resolution']
        self.transform_file_path = camera['transform_file_path']
        
        # Point cloud settings
        pc = base['point_cloud']
        self.require_both_cameras = pc['require_both_cameras']
        self.voxel_size = pc['voxel_size']
        self.workspace_bounds = pc['workspace_bounds']
        self.distance_threshold = pc['distance_threshold']
        self.publish_point_clouds = pc['publish_point_clouds']
        
        # Pose estimation settings
        pose = base['pose_estimation']
        self.pose_method = pose['method']
        
        # Shared settings
        shared = pose['shared']
        self.shared = shared
        self.poisson_reconstruction = shared['poisson_reconstruction']
        self.outlier_removal = shared['outlier_removal']
        self.voxel_downsample_size = shared['voxel_downsample_size']
        self.publish_poses = shared['publish_poses']
        self.publish_transforms = shared['publish_transforms']
        
        # Shared visualization settings
        self.visualize_fused_workspace = shared.get('visualize_fused_workspace', False)
        self.enable_detected_object_clouds_visualization = shared.get('enable_detected_object_clouds_visualization', False)
        
        # Method-specific settings
        if self.pose_method == 'icp':
            icp = pose['icp']
            self.model_folder_path = icp['model_folder_path']
            self.icp_distance_threshold = icp['distance_threshold']
            self.icp_max_iterations = icp['max_iterations']
            self.icp_convergence_threshold = icp['convergence_threshold']
            self.align_to_world_frame = icp.get('align_to_world_frame', False)
            self.visualize_icp_alignment = icp['visualize_icp_alignment']
            self.visualize_grasp_poses = icp['visualize_grasp_poses']
            
            # ICP grasp planning
            self.icp_grasp_planning_enabled = icp.get('icp_grasp_planning_enabled', True)
            self.max_grasp_candidates = icp.get('max_grasp_candidates', 100)
            
            # Set default values for superquadric properties (in case they're accessed)
            self.outlier_ratio = 0.999
            self.use_kmeans_clustering = False
            self.sq_max_iterations = 1000
            self.sq_convergence_threshold = 1e-6
            self.gripper_jaw_length = 0.037
            self.gripper_max_opening = 0.08
            
            # Default superquadric visualization settings (all disabled for ICP)
            self.enable_superquadric_fit_visualization = False
            self.enable_all_valid_grasps_visualization = False
            self.enable_best_grasps_visualization = False
            self.enable_support_test_visualization = False
            self.enable_collision_test_visualization = False
            
        else:  # superquadric method
            sq = pose['superquadric']
            self.outlier_ratio = sq['outlier_ratio']
            self.use_kmeans_clustering = sq['use_kmeans_clustering']
            self.sq_max_iterations = sq['max_iterations']
            self.sq_convergence_threshold = sq['convergence_threshold']
            self.gripper_jaw_length = sq['gripper_jaw_length']
            self.gripper_max_opening = sq['gripper_max_opening']
            
            # Superquadric grasp planning
            self.sq_grasp_planning_enabled = sq.get('sq_grasp_planning_enabled', True)

            # Superquadric visualizations
            self.enable_superquadric_fit_visualization = sq['enable_superquadric_fit_visualization']
            self.enable_all_valid_grasps_visualization = sq['enable_all_valid_grasps_visualization']
            self.enable_best_grasps_visualization = sq['enable_best_grasps_visualization']
            self.enable_support_test_visualization = sq['enable_support_test_visualization']
            self.enable_collision_test_visualization = sq['enable_collision_test_visualization']
            
            # Set default values for ICP properties (in case they're accessed)
            self.model_folder_path = ""
            self.icp_distance_threshold = 0.03
            self.icp_max_iterations = 50
            self.icp_convergence_threshold = 1e-6
            self.visualize_icp_alignment = False
            self.visualize_grasp_poses = False
            self.max_grasp_candidates = 100
        
        # Web interface
        web = base.get('web_interface', {})
        self.web_enabled = web.get('enabled', False)
        self.web_port = web.get('port', 8080)
        self.web_dir = web.get('web_dir', '/tmp/grasp_system_live')
        self.web_update_rate = web.get('update_rate', 15.0)
        self.web_image_quality = web.get('image_quality', 85)
        

    def _determine_device(self, device_str: str) -> str:
        """Determine device based on availability"""
        if device_str == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_str
    
    # Helper methods for getting config subsets (if still needed)
    def get_camera_config(self):
        """Get camera-specific config"""
        return {
            'camera1_sn': self.camera1_sn,
            'camera2_sn': self.camera2_sn,
            'resolution': self.camera_resolution,
            'transform_file_path': self.transform_file_path,
            'device': self.device
        }
    
    def get_detection_config(self):
        """Get detection-specific config"""
        return {
            'method': self.detection_method,
            'yolo_model_path': self.yolo_model_path,
            'confidence_threshold': self.confidence_threshold,
            'classes': self.detection_classes,
            'class_names': self.class_names
        }
    
    def get_point_cloud_config(self):
        """Get point cloud-specific config"""
        return {
            'voxel_size': self.voxel_size,
            'workspace_bounds': self.workspace_bounds,
            'distance_threshold': self.distance_threshold,
            'publish_point_clouds': self.publish_point_clouds,
            'device': self.device
        }
    
    def get(self, key: str, default=None):
        """Dictionary-like access for backwards compatibility"""
        return getattr(self, key, default)

    def __getitem__(self, key):
        """Allow dict-style access: config['key']"""
        return getattr(self, key)

    def __contains__(self, key):
        """Allow 'key in config' checks"""
        return hasattr(self, key)