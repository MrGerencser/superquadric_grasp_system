# utils/ros_publisher.py
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R_simple
import numpy as np

class ROSPublisher:
    """Handles ROS publishing for pose estimation results"""
    
    def __init__(self, node, config):
        self.node = node
        self.config = config
        self.logger = node.get_logger()
        
        # Check if publishing is enabled
        self.publish_poses = config.get('publish_poses', True)
        self.publish_transforms = config.get('publish_transforms', True)
        self.target_frame = config.get('target_frame', 'panda_link0')
        
        self.pose_publisher = None
        self.tf_broadcaster = None
    
    def initialize(self) -> bool:
        """Initialize ROS publishers"""
        try:
            if self.publish_poses:
                self.pose_publisher = self.node.create_publisher(
                    PoseStamped, '/perception/object_pose', 1
                )
                
            if self.publish_transforms:
                self.tf_broadcaster = TransformBroadcaster(self.node)
                
            self.logger.info("ROS publishers initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS publishers: {e}")
            return False
    
    def publish_pose(self, pose_matrix, class_id, object_id):
        """Publish pose and transform messages"""
        try:
            if self.publish_poses and self.pose_publisher:
                pose_msg = self._matrix_to_pose_stamped(pose_matrix, class_id, object_id)
                self.pose_publisher.publish(pose_msg)
                
            if self.publish_transforms and self.tf_broadcaster:
                transform_msg = self._create_transform_stamped(pose_matrix, class_id, object_id)
                self.tf_broadcaster.sendTransform(transform_msg)
                
        except Exception as e:
            self.logger.error(f"Failed to publish pose: {e}")
    
    def _matrix_to_pose_stamped(self, transformation_matrix: np.ndarray, 
                               class_id: int, object_id: int) -> PoseStamped:
        """Convert transformation matrix to PoseStamped message"""
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.target_frame
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        
        # Extract position
        pose_msg.pose.position.x = float(transformation_matrix[0, 3])
        pose_msg.pose.position.y = float(transformation_matrix[1, 3])
        pose_msg.pose.position.z = float(transformation_matrix[2, 3])
        
        # Extract rotation and convert to quaternion
        rotation_matrix = transformation_matrix[:3, :3]
        r = R_simple.from_matrix(rotation_matrix)
        quat = r.as_quat()  # [x, y, z, w]
        
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        
        return pose_msg
    
    def _create_transform_stamped(self, transformation_matrix: np.ndarray, 
                                 class_id: int, object_id: int) -> TransformStamped:
        """Create TransformStamped message from transformation matrix"""
        transform = TransformStamped()
        transform.header.stamp = self.node.get_clock().now().to_msg()
        transform.header.frame_id = self.target_frame
        transform.child_frame_id = f"object_{class_id}_{object_id}"
        
        # Position
        transform.transform.translation.x = float(transformation_matrix[0, 3])
        transform.transform.translation.y = float(transformation_matrix[1, 3])
        transform.transform.translation.z = float(transformation_matrix[2, 3])
        
        # Rotation
        rotation_matrix = transformation_matrix[:3, :3]
        r = R_simple.from_matrix(rotation_matrix)
        quat = r.as_quat()
        
        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])
        
        return transform