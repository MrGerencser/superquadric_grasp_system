import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation as R_simple


def create_pose_message(position: List[float], 
                       orientation: List[float], 
                       frame_id: str = "base_link", 
                       stamp: Optional[object] = None) -> PoseStamped:
    """
    Create a PoseStamped message from position and orientation
    
    Args:
        position: [x, y, z] position
        orientation: [x, y, z, w] quaternion or [roll, pitch, yaw] euler angles
        frame_id: Reference frame
        stamp: Timestamp (if None, uses current time)
    
    Returns:
        PoseStamped message
    """
    pose_msg = PoseStamped()
    
    # Header
    pose_msg.header.frame_id = frame_id
    if stamp is not None:
        pose_msg.header.stamp = stamp
    # If stamp is None, ROS will use current time when published
    
    # Position
    pose_msg.pose.position.x = float(position[0])
    pose_msg.pose.position.y = float(position[1])
    pose_msg.pose.position.z = float(position[2])
    
    # Orientation
    if len(orientation) == 4:
        # Quaternion [x, y, z, w]
        pose_msg.pose.orientation.x = float(orientation[0])
        pose_msg.pose.orientation.y = float(orientation[1])
        pose_msg.pose.orientation.z = float(orientation[2])
        pose_msg.pose.orientation.w = float(orientation[3])
    elif len(orientation) == 3:
        # Euler angles [roll, pitch, yaw] - convert to quaternion
        r = R_simple.from_euler('xyz', orientation)
        quat = r.as_quat()  # Returns [x, y, z, w]
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
    else:
        raise ValueError("Orientation must be either 3 (euler) or 4 (quaternion) elements")
    
    return pose_msg


def load_camera_transforms(transform_file_path: str) -> Dict:
    """
    Load camera transformation matrices from YAML file
    
    Args:
        transform_file_path: Path to transform YAML file
        
    Returns:
        Dictionary containing camera transforms
    """
    try:
        with open(transform_file_path, 'r') as file:
            transforms = yaml.safe_load(file)
        return transforms
    except FileNotFoundError:
        # Return default transforms if file not found
        return {
            'camera1': {
                'rotation': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                'translation': [0.0, 0.0, 0.0]
            },
            'camera2': {
                'rotation': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                'translation': [0.5, 0.0, 0.0]
            }
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load camera transforms: {e}")


def transform_point_cloud(point_cloud: np.ndarray,
                         rotation_matrix: np.ndarray,
                         translation_vector: np.ndarray) -> np.ndarray:
    """
    Transform point cloud using rotation matrix and translation vector
    
    Args:
        point_cloud: Nx3 array of points
        rotation_matrix: 3x3 rotation matrix
        translation_vector: 3x1 translation vector
        
    Returns:
        Transformed point cloud
    """
    # Apply rotation and translation
    transformed_points = (rotation_matrix @ point_cloud.T).T + translation_vector
    return transformed_points


def create_transform_matrix(rotation: List[float], 
                           translation: List[float]) -> np.ndarray:
    """
    Create 4x4 transformation matrix from rotation and translation
    
    Args:
        rotation: 9-element list representing 3x3 rotation matrix (row-major)
        translation: 3-element list [x, y, z]
        
    Returns:
        4x4 transformation matrix
    """
    transform = np.eye(4)
    
    # Set rotation part (3x3)
    rotation_matrix = np.array(rotation).reshape(3, 3)
    transform[:3, :3] = rotation_matrix
    
    # Set translation part
    transform[:3, 3] = translation
    
    return transform


def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
    """
    Convert quaternion to rotation matrix
    
    Args:
        quaternion: [x, y, z, w] quaternion
        
    Returns:
        3x3 rotation matrix
    """
    r = R_simple.from_quat(quaternion)
    return r.as_matrix()


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> List[float]:
    """
    Convert rotation matrix to quaternion
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        [x, y, z, w] quaternion
    """
    r = R_simple.from_matrix(rotation_matrix)
    return r.as_quat().tolist()


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    """
    Convert Euler angles to quaternion
    
    Args:
        roll, pitch, yaw: Euler angles in radians
        
    Returns:
        [x, y, z, w] quaternion
    """
    r = R_simple.from_euler('xyz', [roll, pitch, yaw])
    return r.as_quat().tolist()


def quaternion_to_euler(quaternion: List[float]) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles
    
    Args:
        quaternion: [x, y, z, w] quaternion
        
    Returns:
        (roll, pitch, yaw) in radians
    """
    r = R_simple.from_quat(quaternion)
    return tuple(r.as_euler('xyz'))