import numpy as np
import pyzed.sl as sl
import cv2
import torch
from typing import Tuple, Optional
from .base_manager import BaseManager

from ..utils.point_cloud_utils import *

class CameraManager(BaseManager):
    """Manages ZED stereo cameras"""
    
    def __init__(self, node, config):
        super().__init__(node, config)
        self.zed1 = None
        self.zed2 = None
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Camera parameters
        self.camera1_sn = config.get('camera1_sn', 33137761)
        self.camera2_sn = config.get('camera2_sn', 32689769)
        
        # Calibration parameters
        self.fx1 = self.fy1 = self.cx1 = self.cy1 = None
        self.fx2 = self.fy2 = self.cx2 = self.cy2 = None
        
        # Transform matrices
        self.rotation1_torch = None
        self.origin1_torch = None
        self.rotation2_torch = None
        self.origin2_torch = None
        
        # Image containers
        self.image1 = None
        self.depth1 = None
        self.image2 = None
        self.depth2 = None
        self.point_cloud1_ws = None
        self.point_cloud2_ws = None
        
    def initialize(self) -> bool:
        """Initialize ZED cameras"""
        try:
            self.logger.info("Initializing ZED cameras...")
            
            # Initialize camera objects
            self.zed1 = sl.Camera()
            self.zed2 = sl.Camera()
            
            # Configure cameras
            if not self._configure_cameras():
                return False
                
            # Load transformations
            if not self._load_transformations():
                return False
                
            # Initialize image containers
            self._initialize_containers()
            
            self.is_initialized = True
            self.logger.info("ZED cameras initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cameras: {e}")
            return False
    
    def _configure_cameras(self) -> bool:
        """Configure and open both cameras"""
        try:
            # Get resolution from config
            resolution_str = self.config.get('resolution', 'HD2K')
            resolution_map = {
                'HD720': sl.RESOLUTION.HD720,
                'HD1080': sl.RESOLUTION.HD1080, 
                'HD2K': sl.RESOLUTION.HD2K
            }
            camera_resolution = resolution_map.get(resolution_str, sl.RESOLUTION.HD2K)
            
            # Camera 1 configuration
            init_params1 = sl.InitParameters()
            init_params1.set_from_serial_number(self.camera1_sn)
            init_params1.camera_resolution = camera_resolution
            init_params1.camera_fps = 10
            init_params1.depth_mode = sl.DEPTH_MODE.NEURAL
            init_params1.depth_minimum_distance = 0.4
            init_params1.coordinate_units = sl.UNIT.METER
            
            # Camera 2 configuration
            init_params2 = sl.InitParameters()
            init_params2.set_from_serial_number(self.camera2_sn)
            init_params2.camera_resolution = camera_resolution
            init_params2.camera_fps = 10
            init_params2.depth_mode = sl.DEPTH_MODE.NEURAL
            init_params2.depth_minimum_distance = 0.4
            init_params2.coordinate_units = sl.UNIT.METER
            
            self.logger.info(f"Using camera resolution: {resolution_str}")
            
            # Open cameras
            err1 = self.zed1.open(init_params1)
            if err1 != sl.ERROR_CODE.SUCCESS:
                self.logger.error(f"Error opening ZED camera 1: {err1}")
                return False
                
            err2 = self.zed2.open(init_params2)
            if err2 != sl.ERROR_CODE.SUCCESS:
                self.logger.error(f"Error opening ZED camera 2: {err2}")
                self.zed1.close()
                return False
            
            # Get calibration parameters
            self._get_calibration_parameters()
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring cameras: {e}")
            return False
    
    def _get_calibration_parameters(self):
        """Extract camera calibration parameters"""
        calib_params1 = self.zed1.get_camera_information().camera_configuration.calibration_parameters
        self.fx1, self.fy1 = calib_params1.left_cam.fx, calib_params1.left_cam.fy
        self.cx1, self.cy1 = calib_params1.left_cam.cx, calib_params1.left_cam.cy
        
        calib_params2 = self.zed2.get_camera_information().camera_configuration.calibration_parameters
        self.fx2, self.fy2 = calib_params2.left_cam.fx, calib_params2.left_cam.fy
        self.cx2, self.cy2 = calib_params2.left_cam.cx, calib_params2.left_cam.cy
        
        self.logger.info(f"Camera 1 calibration: fx={self.fx1}, fy={self.fy1}, cx={self.cx1}, cy={self.cy1}")
        self.logger.info(f"Camera 2 calibration: fx={self.fx2}, fy={self.fy2}, cx={self.cx2}, cy={self.cy2}")
    
    def _load_transformations(self) -> bool:
        """Load camera transformation matrices"""
        try:
            import yaml
            transform_file = self.config.get('transform_file_path')
            
            with open(transform_file, 'r') as file:
                transforms = yaml.safe_load(file)
            
            # Extract transformation matrices
            T_chess_cam1 = np.array(transforms['transforms']['T_chess_cam1'])
            T_chess_cam2 = np.array(transforms['transforms']['T_chess_cam2'])
            T_robot_chess = np.array(transforms['transforms']['T_robot_chess'])
            
            # Calculate robot-to-camera transforms
            T_robot_cam1 = T_robot_chess @ T_chess_cam1
            T_robot_cam2 = T_robot_chess @ T_chess_cam2
            
            # Convert to torch tensors
            self.rotation1_torch = torch.tensor(T_robot_cam1[:3, :3], dtype=torch.float32, device=self.device)
            self.origin1_torch = torch.tensor(T_robot_cam1[:3, 3], dtype=torch.float32, device=self.device)
            self.rotation2_torch = torch.tensor(T_robot_cam2[:3, :3], dtype=torch.float32, device=self.device)
            self.origin2_torch = torch.tensor(T_robot_cam2[:3, 3], dtype=torch.float32, device=self.device)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load transformations: {e}")
            return False
    
    def _initialize_containers(self):
        """Initialize image and depth containers"""
        self.image1 = sl.Mat()
        self.depth1 = sl.Mat()
        self.image2 = sl.Mat()
        self.depth2 = sl.Mat()
        
        # Get resolution from config and map to actual dimensions
        resolution_str = self.config.get('resolution', 'HD2K')
        resolution_dims = {
            'HD720': (1280, 720),
            'HD1080': (1920, 1080),
            'HD2K': (2208, 1242)
        }
        
        width, height = resolution_dims.get(resolution_str, (2208, 1242))
        
        # Scale down for point cloud processing (optional optimization)
        pc_width, pc_height = width // 2, height // 2
        
        resolution = sl.Resolution(pc_width, pc_height)
        self.point_cloud1_ws = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        self.point_cloud2_ws = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        
        self.logger.info(f"Initialized containers for resolution: {resolution_str} ({width}x{height})")
    
    def capture_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture frames from both cameras"""
        try:
            if not self.is_ready():
                return None, None
            
            # Grab frames
            if (self.zed1.grab() != sl.ERROR_CODE.SUCCESS or 
                self.zed2.grab() != sl.ERROR_CODE.SUCCESS):
                self.logger.warn("Failed to grab frames from cameras")
                return None, None
            
            # Retrieve images
            self.zed1.retrieve_image(self.image1, view=sl.VIEW.LEFT)
            self.zed2.retrieve_image(self.image2, view=sl.VIEW.LEFT)
            
            # Convert to OpenCV format
            frame1 = cv2.cvtColor(self.image1.get_data(), cv2.COLOR_BGRA2BGR)
            frame2 = cv2.cvtColor(self.image2.get_data(), cv2.COLOR_BGRA2BGR)
            
            return frame1, frame2
            
        except Exception as e:
            self.logger.error(f"Error capturing frames: {e}")
            return None, None
    
    def get_depth_maps(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get depth maps from both cameras"""
        try:
            depth_result1 = self.zed1.retrieve_measure(self.depth1, measure=sl.MEASURE.DEPTH)
            depth_result2 = self.zed2.retrieve_measure(self.depth2, measure=sl.MEASURE.DEPTH)
            
            if depth_result1 != sl.ERROR_CODE.SUCCESS or depth_result2 != sl.ERROR_CODE.SUCCESS:
                return None, None
                
            return self.depth1.get_data(), self.depth2.get_data()
            
        except Exception as e:
            self.logger.error(f"Error retrieving depth maps: {e}")
            return None, None
    
    def get_point_clouds(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        try:
            # Retrieve point clouds from ZED
            self.zed1.retrieve_measure(self.point_cloud1_ws, measure=sl.MEASURE.XYZ)
            self.zed2.retrieve_measure(self.point_cloud2_ws, measure=sl.MEASURE.XYZ)
            
            # Convert to numpy arrays (keep only XYZ coordinates)
            pc1_data = self.point_cloud1_ws.get_data()[:, :, :3]
            pc2_data = self.point_cloud2_ws.get_data()[:, :, :3]
            
            # Reshape and filter valid points
            pc1_points = pc1_data.reshape(-1, 3)
            pc2_points = pc2_data.reshape(-1, 3)
            
            # Filter out invalid points (NaN, inf, zeros)
            valid_mask1 = np.isfinite(pc1_points).all(axis=1) & (np.linalg.norm(pc1_points, axis=1) > 0)
            valid_mask2 = np.isfinite(pc2_points).all(axis=1) & (np.linalg.norm(pc2_points, axis=1) > 0)
            
            pc1_filtered = pc1_points[valid_mask1]
            pc2_filtered = pc2_points[valid_mask2]
            
            # Create Open3D point clouds
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(pc1_filtered)
            
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pc2_filtered)
            
            return pcd1, pcd2
            
        except Exception as e:
            self.logger.error(f"Error getting point clouds: {e}")
            return None, None
    
    def is_ready(self) -> bool:
        """Check if cameras are ready"""
        return (self.is_initialized and 
                self.zed1 is not None and 
                self.zed2 is not None)
    
    def cleanup(self):
        """Clean up camera resources"""
        try:
            if self.zed1:
                self.zed1.close()
            if self.zed2:
                self.zed2.close()
            self.logger.info("Cameras cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during camera cleanup: {e}")