import numpy as np
import torch
import open3d as o3d
from typing import Tuple, List, Optional
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from .base_manager import BaseManager
from .camera_manager import CameraManager
from ..utils.point_cloud_utils import PointCloudProcessor

class PointCloudManager(BaseManager):
    """Manages point cloud processing and fusion"""
    
    def __init__(self, node, config):
        super().__init__(node, config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.voxel_size = config.get('voxel_size', 0.003)
        self.workspace_bounds = config.get('workspace_bounds', [-0.25, 0.75, -0.5, 0.5, -0.05, 2.0])
        self.distance_threshold = config.get('distance_threshold', 0.3)
        self.require_both_cameras = config.get('require_both_cameras', True)
        
        # Publishing settings
        self.publish_point_clouds = config.get('publish_point_clouds', True)
        shared_config = config.get('shared', {})
        self.point_cloud_processor = PointCloudProcessor(shared_config)
        
        # ROS publishers (will be initialized in initialize())
        self.fused_workspace_publisher = None
        self.fused_objects_publisher = None
        self.subtracted_cloud_publisher = None
        self.visualize_fused_workspace = config.get('visualize_fused_workspace', False)

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
                                depth_np1, depth_np2, camera_manager: CameraManager, fused_workspace_np: np.ndarray) -> Tuple[List, List, np.ndarray, np.ndarray]:
        """Extract object point clouds from detection masks and optionally publish them"""
        try:
            # Clear GPU cache at start
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Get camera parameters directly from camera_manager
            fx1, fy1, cx1, cy1 = camera_manager.fx1, camera_manager.fy1, camera_manager.cx1, camera_manager.cy1
            fx2, fy2, cx2, cy2 = camera_manager.fx2, camera_manager.fy2, camera_manager.cx2, camera_manager.cy2
            rotation1_torch = camera_manager.rotation1_torch
            origin1_torch = camera_manager.origin1_torch
            rotation2_torch = camera_manager.rotation2_torch
            origin2_torch = camera_manager.origin2_torch
            
            point_clouds_camera1 = []
            point_clouds_camera2 = []
            
            # Process camera 1 masks with memory management
            if results1.masks is not None and results1.masks.data.numel() > 0:
                depth_map1_torch = torch.tensor(depth_np1, dtype=torch.float32, device='cpu')
                if self.device == 'cuda':
                    depth_map1_torch = depth_map1_torch.to(self.device)
                
                try:
                    for i, individual_mask_tensor in enumerate(results1.masks.data):
                        mask_indices = torch.nonzero(individual_mask_tensor, as_tuple=False)

                        if mask_indices.numel() > 0: 
                            with torch.amp.autocast('cuda'):
                                points_3d = self.convert_mask_to_3d_points(
                                    mask_indices, depth_map1_torch, 
                                    cx1, cy1, fx1, fy1
                                )
                            
                            if points_3d.size(0) > 0:
                                transformed = torch.mm(points_3d, rotation1_torch.T) + origin1_torch
                                
                                # Move to CPU immediately to free GPU memory
                                transformed_np = transformed.cpu().numpy()
                                if self._is_object_in_workspace(transformed_np):
                                    point_clouds_camera1.append((transformed_np, int(class_ids1[i])))
                                
                                # Clean up GPU tensors
                                del points_3d, transformed
                                if self.device == 'cuda':
                                    torch.cuda.empty_cache()
                finally:
                    # Clean up depth tensor
                    del depth_map1_torch
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            # Process camera 2 masks with memory management
            if results2.masks is not None and results2.masks.data.numel() > 0:
                depth_map2_torch = torch.tensor(depth_np2, dtype=torch.float32, device='cpu')
                if self.device == 'cuda':
                    depth_map2_torch = depth_map2_torch.to(self.device)
                
                try:
                    for i, individual_mask_tensor in enumerate(results2.masks.data):
                        mask_indices = torch.nonzero(individual_mask_tensor, as_tuple=False)

                        if mask_indices.numel() > 0: 
                            with torch.amp.autocast('cuda'):
                                points_3d = self.point_cloud_processor.convert_mask_to_3d_points(
                                    mask_indices, depth_map2_torch,
                                    cx2, cy2, fx2, fy2
                                )
                            
                            if points_3d.size(0) > 0:
                                transformed = torch.mm(points_3d, rotation2_torch.T) + origin2_torch
                                
                                transformed_np = transformed.cpu().numpy()
                                if self._is_object_in_workspace(transformed_np):
                                    point_clouds_camera2.append((transformed_np, int(class_ids2[i])))
                                
                                # Clean up GPU tensors
                                del points_3d, transformed
                                if self.device == 'cuda':
                                    torch.cuda.empty_cache()
                finally:
                    # Clean up depth tensor
                    del depth_map2_torch
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()

            # Fuse object point clouds
            _, _, fused_objects = self.point_cloud_processor.fuse_point_clouds_centroid(
                point_clouds_camera1, 
                point_clouds_camera2, 
                distance_threshold=self.distance_threshold,
                require_both_cameras=self.require_both_cameras
            )
            
            fused_object_points = [pc for pc, _ in fused_objects]
            fused_object_classes = [cls for _, cls in fused_objects]
            
            fused_objects_np = (np.vstack(fused_object_points) 
                            if fused_object_points else np.empty((0, 3)))
            
            # DISABLE DUE TO COMPUTATIONAL COST
            # Subtract objects from workspace
            # subtracted_cloud = subtract_point_clouds_gpu(
            #     fused_workspace_np, fused_objects_np, distance_threshold=0.06
            # )
            subtracted_cloud = np.empty((0, 3))  # Empty array for now

            
            # Publish point clouds if enabled
            if self.publish_point_clouds:
                self.publish_object_clouds(fused_objects_np)
                self.publish_subtracted_cloud(subtracted_cloud)

            return fused_object_points, fused_object_classes, fused_objects_np, subtracted_cloud

        except Exception as e:
            # Clean up on error
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            self.logger.error(f"Error extracting object point clouds: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return empty results with correct tuple structure
            return [], [], np.empty((0, 3)), np.empty((0, 3))

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
        """Check if object is within workspace bounds"""
        try:
            if len(object_points) == 0:
                return False
            
            # Extract workspace bounds
            x_min, x_max = self.workspace_bounds[0], self.workspace_bounds[1]
            y_min, y_max = self.workspace_bounds[2], self.workspace_bounds[3]
            z_min, z_max = self.workspace_bounds[4], self.workspace_bounds[5]
            
            # Check which points are within workspace bounds
            x_in_bounds = (object_points[:, 0] >= x_min) & (object_points[:, 0] <= x_max)
            y_in_bounds = (object_points[:, 1] >= y_min) & (object_points[:, 1] <= y_max)
            z_in_bounds = (object_points[:, 2] >= z_min) & (object_points[:, 2] <= z_max)
            
            points_in_workspace = x_in_bounds & y_in_bounds & z_in_bounds
            coverage = np.sum(points_in_workspace) / len(object_points)
            
            # Check object center position
            object_center = np.mean(object_points, axis=0)
            center_in_workspace = (
                x_min <= object_center[0] <= x_max and
                y_min <= object_center[1] <= y_max and
                z_min <= object_center[2] <= z_max
            )
            
            return (center_in_workspace and coverage >= coverage_threshold) or (coverage >= 0.8)
            
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