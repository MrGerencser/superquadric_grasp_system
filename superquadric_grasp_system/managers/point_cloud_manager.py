import numpy as np
import torch
import open3d as o3d
from typing import Tuple, List, Optional
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

from .camera_manager import CameraManager
from ..utils.point_cloud_utils import PointCloudProcessor

class PointCloudManager:
    """Manages point cloud processing and fusion"""
    
    def __init__(self, node, device: str, voxel_size: float,
                 workspace_bounds: list, distance_threshold: float,
                 require_both_cameras: bool, publish_enabled: bool,
                 poisson_reconstruction: bool = False, outlier_removal: bool = True,
                 voxel_downsample_size: float = 0.002, visualize_fused_workspace: bool = False,
                 enable_detected_object_clouds_visualization: bool = False):
        self.node = node
        self.logger = node.get_logger()
        
        # Configuration
        self.device = device
        self.voxel_size = voxel_size
        self.workspace_bounds = workspace_bounds
        self.distance_threshold = distance_threshold
        self.require_both_cameras = require_both_cameras
        self.publish_point_clouds = publish_enabled
        self.visualize_fused_workspace = visualize_fused_workspace
        
        # Create point cloud processor with visualization flag
        self.point_cloud_processor = PointCloudProcessor(
            poisson_reconstruction=poisson_reconstruction,
            outlier_removal=outlier_removal,
            voxel_downsample_size=voxel_downsample_size,
            enable_detected_object_clouds_visualization=enable_detected_object_clouds_visualization
        )
        
        # ROS publishers (will be initialized in initialize())
        self.fused_workspace_publisher = None
        self.fused_objects_publisher = None
        self.subtracted_cloud_publisher = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """Initialize point cloud manager and ROS publishers"""
        try:
            # Initialize ROS publishers if publishing is enabled
            if self.publish_point_clouds:
                self.fused_workspace_publisher = self.node.create_publisher(
                    PointCloud2, '/perception/fused_workspace_cloud', 1)
                self.fused_objects_publisher = self.node.create_publisher(
                    PointCloud2, '/perception/fused_objects_cloud', 1)
                self.subtracted_cloud_publisher = self.node.create_publisher(
                    PointCloud2, '/perception/subtracted_cloud', 1)
                self.logger.info("Point cloud publishers initialized")
            
            self.is_initialized = True
            self.logger.info("Point cloud manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize point cloud manager: {e}")
            return False
        
    
    def process_point_clouds(self, camera_manager: CameraManager):
        """Process point clouds from cameras and return fused workspace data"""
        try:
            # Get point clouds from cameras as Open3D PointClouds
            pcd1, pcd2 = camera_manager.get_point_clouds()
            
            if pcd1 is None or pcd2 is None:
                self.logger.error("Failed to get point clouds from cameras")
                return None, None, None, None
            
            # Convert to numpy for transformation
            pc1_np = np.asarray(pcd1.points)
            pc2_np = np.asarray(pcd2.points)
            
            # Get camera parameters for transformations
            rotation1_torch = camera_manager.rotation1_torch
            origin1_torch = camera_manager.origin1_torch
            rotation2_torch = camera_manager.rotation2_torch
            origin2_torch = camera_manager.origin2_torch
            
            # Transform to robot frame (using numpy operations)
            rotation1_np = rotation1_torch.cpu().numpy()
            origin1_np = origin1_torch.cpu().numpy()
            rotation2_np = rotation2_torch.cpu().numpy()
            origin2_np = origin2_torch.cpu().numpy()
            
            pc1_transformed = (pc1_np @ rotation1_np.T) + origin1_np
            pc2_transformed = (pc2_np @ rotation2_np.T) + origin2_np
            
            # Crop to workspace bounds using numpy
            x_bounds = (self.workspace_bounds[0], self.workspace_bounds[1])
            y_bounds = (self.workspace_bounds[2], self.workspace_bounds[3])
            z_bounds = (self.workspace_bounds[4], self.workspace_bounds[5])
            
            # Create masks for workspace bounds
            mask1 = ((pc1_transformed[:, 0] >= x_bounds[0]) & (pc1_transformed[:, 0] <= x_bounds[1]) &
                    (pc1_transformed[:, 1] >= y_bounds[0]) & (pc1_transformed[:, 1] <= y_bounds[1]) &
                    (pc1_transformed[:, 2] >= z_bounds[0]) & (pc1_transformed[:, 2] <= z_bounds[1]))
            
            mask2 = ((pc2_transformed[:, 0] >= x_bounds[0]) & (pc2_transformed[:, 0] <= x_bounds[1]) &
                    (pc2_transformed[:, 1] >= y_bounds[0]) & (pc2_transformed[:, 1] <= y_bounds[1]) &
                    (pc2_transformed[:, 2] >= z_bounds[0]) & (pc2_transformed[:, 2] <= z_bounds[1]))
            
            pc1_cropped = pc1_transformed[mask1]
            pc2_cropped = pc2_transformed[mask2]
            
            # Fuse workspace point clouds
            fused_workspace_np = np.vstack([pc1_cropped, pc2_cropped])
            
            # If fused_workspace is empty log a warning
            if fused_workspace_np.size == 0:
                self.logger.warning("Fused workspace point cloud is empty after cropping")
            
            # Create Open3D point cloud for the fused result
            pcd_fused_workspace = o3d.geometry.PointCloud()
            pcd_fused_workspace.points = o3d.utility.Vector3dVector(fused_workspace_np)
            
            # Optional visualization
            if self.visualize_fused_workspace:
                o3d.visualization.draw_geometries([pcd_fused_workspace], window_name="Fused Workspace Point Cloud")
            
            # Publish workspace cloud if enabled
            if self.publish_point_clouds:
                self.publish_workspace_cloud(fused_workspace_np)
                
            return pc1_cropped, pc2_cropped, fused_workspace_np, pcd_fused_workspace
            
        except Exception as e:
            self.logger.error(f"Error processing point clouds: {e}")
            return None, None, None, None
    
    def extract_object_point_clouds(self, results1, results2, class_ids1, class_ids2,
                            depth_np1, depth_np2, camera_manager: CameraManager, fused_workspace_np: np.ndarray):
        """Extract object point clouds with CPU-only processing for speed"""
        try:
            # OPTIMIZATION 1: Skip GPU entirely for small masks
            point_clouds_camera1 = []
            point_clouds_camera2 = []
            
            # Get camera parameters as numpy arrays (avoid GPU)
            rotation1_np = camera_manager.rotation1_torch.cpu().numpy()
            origin1_np = camera_manager.origin1_torch.cpu().numpy()
            rotation2_np = camera_manager.rotation2_torch.cpu().numpy()
            origin2_np = camera_manager.origin2_torch.cpu().numpy()
            
            # Process camera 1 masks with CPU only
            if results1.masks is not None and results1.masks.data.numel() > 0:
                for i, individual_mask_tensor in enumerate(results1.masks.data):
                    # Convert mask to CPU numpy immediately
                    mask_np = individual_mask_tensor.cpu().numpy()
                    mask_indices_np = np.argwhere(mask_np)
                    
                    if len(mask_indices_np) > 0:
                        # CPU-based 3D point conversion
                        points_3d_np = self._convert_mask_to_3d_points_cpu(
                            mask_indices_np, depth_np1, 
                            camera_manager.cx1, camera_manager.cy1, 
                            camera_manager.fx1, camera_manager.fy1
                        )
                        
                        if len(points_3d_np) > 0:
                            # CPU transformation
                            transformed_np = (points_3d_np @ rotation1_np.T) + origin1_np
                            
                            if self._is_object_in_workspace(transformed_np):
                                point_clouds_camera1.append((transformed_np, int(class_ids1[i])))
            
            # Process camera 2 masks with CPU only
            if results2.masks is not None and results2.masks.data.numel() > 0:
                for i, individual_mask_tensor in enumerate(results2.masks.data):
                    mask_np = individual_mask_tensor.cpu().numpy()
                    mask_indices_np = np.argwhere(mask_np)
                    
                    if len(mask_indices_np) > 0:
                        points_3d_np = self._convert_mask_to_3d_points_cpu(
                            mask_indices_np, depth_np2,
                            camera_manager.cx2, camera_manager.cy2,
                            camera_manager.fx2, camera_manager.fy2
                        )
                        
                        if len(points_3d_np) > 0:
                            transformed_np = (points_3d_np @ rotation2_np.T) + origin2_np
                            
                            if self._is_object_in_workspace(transformed_np):
                                point_clouds_camera2.append((transformed_np, int(class_ids2[i])))
            
            # Continue with existing fusion logic...
            _, _, fused_objects = self.point_cloud_processor.fuse_point_clouds_centroid(
                point_clouds_camera1, point_clouds_camera2, 
                distance_threshold=self.distance_threshold,
                require_both_cameras=self.require_both_cameras
            )
            
            fused_object_points = [pc for pc, _ in fused_objects]
            fused_object_classes = [cls for _, cls in fused_objects]
            fused_objects_np = (np.vstack(fused_object_points) if fused_object_points else np.empty((0, 3)))
            subtracted_cloud = np.empty((0, 3))

            # NEW: Direct visualization call
            if self.point_cloud_processor.enable_detected_object_clouds_visualization and len(fused_object_points) > 0:
                # Set up visualizer if not already done
                if self.point_cloud_processor.visualizer is None:
                    try:
                        from ..visualization.main_visualizer import PerceptionVisualizer
                        self.point_cloud_processor.visualizer = PerceptionVisualizer()
                        self.logger.info("Detected object cloud visualization enabled")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize visualizer: {e}")
                        return fused_object_points, fused_object_classes, fused_objects_np, subtracted_cloud
                
                # Visualize each detected object
                for i, (fused_points, class_id) in enumerate(zip(fused_object_points, fused_object_classes)):
                    # Collect original points for this class from both cameras
                    original_points_for_class = []
                    for pc, cls in point_clouds_camera1:
                        if cls == class_id:
                            original_points_for_class.append(pc)
                    for pc, cls in point_clouds_camera2:
                        if cls == class_id:
                            original_points_for_class.append(pc)
                    
                    if original_points_for_class:
                        original_combined = np.vstack(original_points_for_class)
                        object_center = np.mean(fused_points, axis=0)
                        
                        # Direct visualization call
                        try:
                            self.point_cloud_processor.visualizer.visualize_detected_cloud_filtering(
                                original_points=original_combined,
                                filtered_points=fused_points,
                                class_id=class_id,
                                object_center=object_center,
                                window_name=f"Object {i+1} (Class {class_id}) - Original vs Fused"
                            )
                            self.logger.debug(f"Visualized object {i+1} with class {class_id}")
                        except Exception as viz_error:
                            self.logger.warning(f"Visualization failed for object {i+1}: {viz_error}")

            
            return fused_object_points, fused_object_classes, fused_objects_np, subtracted_cloud

        except Exception as e:
            self.logger.error(f"Error extracting object point clouds: {e}")
            return [], [], np.empty((0, 3)), np.empty((0, 3))
        
    def _convert_mask_to_3d_points_cpu(self, mask_indices_np: np.ndarray, depth_map: np.ndarray, 
                                    cx: float, cy: float, fx: float, fy: float) -> np.ndarray:
        """Convert 2D mask indices to 3D points using CPU operations"""
        try:
            if len(mask_indices_np) == 0:
                return np.empty((0, 3))
            
            # Extract pixel coordinates
            v_coords = mask_indices_np[:, 0]  # row indices
            u_coords = mask_indices_np[:, 1]  # column indices
            
            # Get depth values
            depth_values = depth_map[v_coords, u_coords]
            
            # Filter out invalid depth values
            valid_depth_mask = (depth_values > 0) & (depth_values < 10.0)  # 10m max depth
            
            if not np.any(valid_depth_mask):
                return np.empty((0, 3))
            
            # Keep only valid points
            u_valid = u_coords[valid_depth_mask]
            v_valid = v_coords[valid_depth_mask]
            depth_valid = depth_values[valid_depth_mask]
            
            # Convert to 3D coordinates
            x = (u_valid - cx) * depth_valid / fx
            y = (v_valid - cy) * depth_valid / fy
            z = depth_valid
            
            # Stack into 3D points
            points_3d = np.column_stack([x, y, z])
            
            return points_3d
            
        except Exception as e:
            self.logger.error(f"Error in CPU 3D point conversion: {e}")
            return np.empty((0, 3))

    def publish_workspace_cloud(self, workspace_cloud: np.ndarray):
        """Publish workspace point cloud"""
        try:
            if self.fused_workspace_publisher and workspace_cloud is not None and len(workspace_cloud) > 0:
                header = self._create_header()
                workspace_msg = pc2.create_cloud_xyz32(header, workspace_cloud)
                self.fused_workspace_publisher.publish(workspace_msg)
                self.logger.debug(f"Published workspace cloud with {len(workspace_cloud)} points")
                
        except Exception as e:
            self.logger.error(f"Error publishing workspace cloud: {e}")

    def publish_object_clouds(self, objects_cloud: np.ndarray):
        """Publish object point clouds"""
        try:
            if self.fused_objects_publisher and objects_cloud is not None and len(objects_cloud) > 0:
                header = self._create_header()
                objects_msg = pc2.create_cloud_xyz32(header, objects_cloud)
                self.fused_objects_publisher.publish(objects_msg)
                self.logger.debug(f"Published object clouds with {len(objects_cloud)} points")
                
        except Exception as e:
            self.logger.error(f"Error publishing object clouds: {e}")

    def publish_subtracted_cloud(self, subtracted_cloud: np.ndarray):
        """Publish subtracted point cloud"""
        try:
            if self.subtracted_cloud_publisher and subtracted_cloud is not None and len(subtracted_cloud) > 0:
                header = self._create_header()
                subtracted_msg = pc2.create_cloud_xyz32(header, subtracted_cloud)
                self.subtracted_cloud_publisher.publish(subtracted_msg)
                self.logger.debug(f"Published subtracted cloud with {len(subtracted_cloud)} points")
                
        except Exception as e:
            self.logger.error(f"Error publishing subtracted cloud: {e}")

    def _create_header(self) -> Header:
        """Create ROS header for point cloud messages"""
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = "panda_link0"
        return header
    
    def _is_object_in_workspace(self, object_points: np.ndarray, coverage_threshold: float = 0.5) -> bool:
        """Workspace bounds check"""
        try:
            if len(object_points) == 0:
                return False
            
            bounds = np.array(self.workspace_bounds).reshape(3, 2)  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            
            # Vectorized bounds check
            in_bounds = np.all((object_points >= bounds[:, 0]) & (object_points <= bounds[:, 1]), axis=1)
            coverage = np.mean(in_bounds)
            
            # Quick center check
            center = np.mean(object_points, axis=0)
            center_in_bounds = np.all((center >= bounds[:, 0]) & (center <= bounds[:, 1]))
            
            return center_in_bounds and coverage >= coverage_threshold
            
        except Exception as e:
            self.logger.error(f"Error in workspace bounds check: {e}")
            return False


    
    def is_ready(self) -> bool:
        """Check if point cloud manager is ready"""
        return self.is_initialized
    
    def cleanup(self):
        """Clean up point cloud resources"""
        try:
            # Publishers will be cleaned up automatically by ROS
            self.logger.info("Point cloud manager cleaned up")
        except Exception as e:
            self.logger.error(f"Error during point cloud manager cleanup: {e}")