import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import yaml
import os
from ament_index_python.packages import get_package_share_directory
from typing import Optional, Tuple

from superquadric_grasp_system.managers.camera_manager import CameraManager

class PointCloudNode(Node):
    """Minimal node for fused workspace point cloud capture and publishing"""
    
    def __init__(self):
        super().__init__('pointcloud_node')
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize camera manager
        self.camera_manager = CameraManager(self, self.config['perception']['camera'])
        
        # Extract workspace bounds
        self.workspace_bounds = self.config['perception']['point_cloud'].get(
            'workspace_bounds', [-0.25, 0.75, -0.5, 0.5, -0.05, 2.0]
        )
        
        # Create publisher for fused RGB point cloud
        self.pointcloud_publisher = self.create_publisher(
            PointCloud2, '/perception/fused_workspace_rgb_cloud', 1
        )
        
        # Processing rate
        self.processing_rate = self.config['perception'].get('processing_rate', 10.0)
        
        # Initialize camera
        if not self.camera_manager.initialize():
            self.get_logger().error("Failed to initialize camera manager")
            return
            
        # Create timer for processing
        self.timer = self.create_timer(1.0/self.processing_rate, self.process_and_publish)
        
        self.get_logger().info("PointCloud Node initialized successfully")
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            package_share_directory = get_package_share_directory('superquadric_grasp_system')
            config_file = os.path.join(package_share_directory, 'config', 'perception_config.yaml')
        except Exception:
            config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'perception_config.yaml')

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Determine device
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            config['perception']['device'] = device
            config['perception']['camera']['device'] = device
            
            return config
        except Exception as e:
            self.get_logger().error(f"Failed to load config: {e}")
            raise
    
    def process_and_publish(self):
        """Main processing loop - capture and publish fused RGB point cloud"""
        try:
            # Step 1: Capture frames and point clouds
            frame1, frame2 = self.camera_manager.capture_frames()
            if frame1 is None or frame2 is None:
                return
            
            pcd1, pcd2 = self.camera_manager.get_point_clouds()
            if pcd1 is None or pcd2 is None:
                return
        
            # Step 2: Get RGB colors for point clouds (with camera ID)
            rgb_colors1 = self._extract_rgb_colors(frame1, pcd1, camera_id=1)
            rgb_colors2 = self._extract_rgb_colors(frame2, pcd2, camera_id=2)
        
            # Step 3: Transform point clouds to robot frame
            pc1_transformed, pc2_transformed = self._transform_point_clouds(pcd1, pcd2)
            if pc1_transformed is None or pc2_transformed is None:
                return
        
            # Step 4: Crop to workspace bounds
            pc1_cropped, colors1_cropped = self._crop_to_workspace(pc1_transformed, rgb_colors1)
            pc2_cropped, colors2_cropped = self._crop_to_workspace(pc2_transformed, rgb_colors2)
        
            # Step 5: Fuse point clouds
            fused_points, fused_colors = self._fuse_point_clouds(
                pc1_cropped, colors1_cropped, pc2_cropped, colors2_cropped
            )
        
            if len(fused_points) == 0:
                self.get_logger().warn("No points in fused workspace cloud")
                return
            
            # Visualize the fused point cloud (optional)
            # self._visualize_point_cloud(fused_points, fused_colors)
        
            # Step 6: Publish RGB point cloud
            self._publish_rgb_pointcloud(fused_points, fused_colors)
        
            self.get_logger().debug(f"Published fused RGB point cloud with {len(fused_points)} points")
        
        except Exception as e:
            self.get_logger().error(f"Error in process_and_publish: {e}")
            
    def _visualize_point_cloud(self, points: np.ndarray, colors: np.ndarray):
        """Visualize the point cloud using Open3D"""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Optionally, visualize in a separate thread
            o3d.visualization.draw_geometries([pcd], window_name="Fused Workspace Point Cloud")
        
        except Exception as e:
            self.get_logger().error(f"Error visualizing point cloud: {e}")
    
    def _extract_rgb_colors(self, frame: np.ndarray, pcd: o3d.geometry.PointCloud, camera_id: int = 1) -> np.ndarray:
        """Extract RGB colors for point cloud points from camera frame"""
        try:
            # Get camera intrinsics based on camera ID
            if camera_id == 1:
                fx, fy = self.camera_manager.fx1, self.camera_manager.fy1
                cx, cy = self.camera_manager.cx1, self.camera_manager.cy1
            else:
                fx, fy = self.camera_manager.fx2, self.camera_manager.fy2
                cx, cy = self.camera_manager.cx2, self.camera_manager.cy2
            
            points = np.asarray(pcd.points)
            height, width = frame.shape[:2]
            
            # Project 3D points to 2D image coordinates
            u = (points[:, 0] * fx / points[:, 2] + cx).astype(int)
            v = (points[:, 1] * fy / points[:, 2] + cy).astype(int)
            
            # Filter valid projections
            valid_mask = (
                (u >= 0) & (u < width) & 
                (v >= 0) & (v < height) & 
                (points[:, 2] > 0)
            )
            
            # Initialize colors array
            colors = np.zeros((len(points), 3), dtype=np.float32)
            
            # Extract RGB values for valid points
            valid_u = u[valid_mask]
            valid_v = v[valid_mask]
            
            # Fix the indexing issue - extract colors point by point
            if len(valid_u) > 0:
                # Extract BGR values and convert to RGB, then normalize to [0,1]
                bgr_values = frame[valid_v, valid_u]  # Shape: (N, 3)
                colors[valid_mask] = bgr_values[:, [2, 1, 0]] / 255.0  # BGR to RGB conversion
            
            return colors
            
        except Exception as e:
            self.get_logger().error(f"Error extracting RGB colors: {e}")
            return np.zeros((len(np.asarray(pcd.points)), 3), dtype=np.float32)
    
    def _transform_point_clouds(self, pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Transform point clouds to robot frame"""
        try:
            # Convert to numpy
            pc1_np = np.asarray(pcd1.points)
            pc2_np = np.asarray(pcd2.points)
            
            # Get transformation matrices from camera manager
            rotation1_np = self.camera_manager.rotation1_torch.cpu().numpy()
            origin1_np = self.camera_manager.origin1_torch.cpu().numpy()
            rotation2_np = self.camera_manager.rotation2_torch.cpu().numpy()
            origin2_np = self.camera_manager.origin2_torch.cpu().numpy()
            
            # Transform to robot frame
            pc1_transformed = (pc1_np @ rotation1_np.T) + origin1_np
            pc2_transformed = (pc2_np @ rotation2_np.T) + origin2_np
            
            return pc1_transformed, pc2_transformed
            
        except Exception as e:
            self.get_logger().error(f"Error transforming point clouds: {e}")
            return None, None
    
    def _crop_to_workspace(self, points: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crop point cloud to workspace bounds"""
        try:
            x_min, x_max = self.workspace_bounds[0], self.workspace_bounds[1]
            y_min, y_max = self.workspace_bounds[2], self.workspace_bounds[3]
            z_min, z_max = self.workspace_bounds[4], self.workspace_bounds[5]
            
            # Create mask for workspace bounds
            mask = (
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            )
            
            return points[mask], colors[mask]
            
        except Exception as e:
            self.get_logger().error(f"Error cropping to workspace: {e}")
            return points, colors
    
    def _fuse_point_clouds(self, pc1: np.ndarray, colors1: np.ndarray, 
                          pc2: np.ndarray, colors2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple fusion - just concatenate point clouds"""
        try:
            if len(pc1) == 0 and len(pc2) == 0:
                return np.empty((0, 3)), np.empty((0, 3))
            elif len(pc1) == 0:
                return pc2, colors2
            elif len(pc2) == 0:
                return pc1, colors1
            else:
                fused_points = np.vstack([pc1, pc2])
                fused_colors = np.vstack([colors1, colors2])
                return fused_points, fused_colors
                
        except Exception as e:
            self.get_logger().error(f"Error fusing point clouds: {e}")
            return np.empty((0, 3)), np.empty((0, 3))
    
    def _publish_rgb_pointcloud(self, points: np.ndarray, colors: np.ndarray):
        """Publish RGB point cloud as ROS2 message"""
        try:
            # Create header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "panda_link0"
            
            # Define point cloud fields for XYZRGB
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
            ]
            
            # Convert RGB colors to packed format
            rgb_packed = self._pack_rgb(colors)
            
            # Create structured array
            cloud_data = np.zeros(len(points), dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)
            ])
            
            cloud_data['x'] = points[:, 0]
            cloud_data['y'] = points[:, 1] 
            cloud_data['z'] = points[:, 2]
            cloud_data['rgb'] = rgb_packed
            
            # Create point cloud message
            pointcloud_msg = pc2.create_cloud(header, fields, cloud_data)
            
            # Publish
            self.pointcloud_publisher.publish(pointcloud_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing RGB point cloud: {e}")
    
    def _pack_rgb(self, colors: np.ndarray) -> np.ndarray:
        """Pack RGB colors into uint32 format"""
        try:
            # Convert [0,1] float to [0,255] uint8
            colors_uint8 = (colors * 255).astype(np.uint8)
            
            # Pack RGB into uint32 (assuming RGB format)
            rgb_packed = (
                (colors_uint8[:, 0].astype(np.uint32) << 16) |  # R
                (colors_uint8[:, 1].astype(np.uint32) << 8) |   # G
                (colors_uint8[:, 2].astype(np.uint32))          # B
            )
            
            return rgb_packed
            
        except Exception as e:
            self.get_logger().error(f"Error packing RGB: {e}")
            return np.zeros(len(colors), dtype=np.uint32)
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'camera_manager'):
                self.camera_manager.cleanup()
            self.get_logger().info("PointCloud Node cleaned up")
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
        node = PointCloudNode()
        rclpy.spin(node)
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