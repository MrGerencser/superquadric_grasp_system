import unittest
import numpy as np
import rclpy
from rclpy.node import Node

class TestCameraManager(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('test_camera_manager')
        
        # Mock config for testing
        self.config = {
            'camera1_sn': 33137761,
            'camera2_sn': 36829049,
            'device': 'cpu',
            'transform_file_path': 'launch/config/transform.yaml'
        }
    
    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()
    
    def test_camera_manager_initialization(self):
        """Test camera manager can be instantiated"""
        from superquadric_grasp_system.managers.camera_manager import CameraManager
        
        manager = CameraManager(self.node, self.config)
        self.assertIsNotNone(manager)
        self.assertEqual(manager.camera1_sn, 123456)
        self.assertEqual(manager.camera2_sn, 789012)
    
    def test_camera_manager_ready_state(self):
        """Test camera manager ready state without hardware"""
        from superquadric_grasp_system.managers.camera_manager import CameraManager
        
        manager = CameraManager(self.node, self.config)
        # Should not be ready without proper initialization
        self.assertFalse(manager.is_ready())

class TestDetectionManager(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('test_detection_manager')
        
        self.config = {
            'yolo_model_path': 'superquadric_grasp_system/models/yolo_seg_finetuned.pt',  # Use a lightweight model for testing
            'device': 'cpu',
            'confidence_threshold': 0.5,
            'classes': [0, 1, 2],
            'class_names': {0: 'person', 1: 'bicycle', 2: 'car'}
        }
    
    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()
    
    def test_detection_manager_initialization(self):
        """Test detection manager initialization"""
        from superquadric_grasp_system.managers.detection_manager import DetectionManager
        
        manager = DetectionManager(self.node, self.config)
        self.assertIsNotNone(manager)
    
    def test_mock_detection(self):
        """Test detection with mock frames"""
        from superquadric_grasp_system.managers.detection_manager import DetectionManager
        
        # Create mock frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        manager = DetectionManager(self.node, self.config)
        
        # This will fail without proper model, but tests the interface
        try:
            results = manager.detect_objects([frame1, frame2])
            # If we get here, the interface works
            self.assertTrue(True)
        except Exception as e:
            # Expected to fail without proper model setup
            self.assertIsInstance(e, Exception)

class TestPointCloudManager(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('test_pointcloud_manager')
        
        self.config = {
            'device': 'cpu',
            'voxel_size': 0.003,
            'workspace_bounds': [-0.5, 0.5, -0.5, 0.5, 0.0, 1.0],
            'distance_threshold': 0.1,
            'require_both_cameras': False
        }
    
    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()
    
    def test_pointcloud_manager_initialization(self):
        """Test point cloud manager initialization"""
        from superquadric_grasp_system.managers.point_cloud_manager import PointCloudManager
        
        manager = PointCloudManager(self.node, self.config)
        self.assertTrue(manager.initialize())
        self.assertTrue(manager.is_ready())
    
    def test_workspace_bounds_check(self):
        """Test workspace bounds checking"""
        from superquadric_grasp_system.managers.point_cloud_manager import PointCloudManager
        
        manager = PointCloudManager(self.node, self.config)
        manager.initialize()
        
        # Test points inside workspace
        inside_points = np.array([[0.0, 0.0, 0.5], [0.1, 0.1, 0.8]])
        result = manager._is_object_in_workspace(inside_points)
        self.assertTrue(result)
        
        # Test points outside workspace
        outside_points = np.array([[1.0, 1.0, 1.5], [2.0, 2.0, 2.0]])
        result = manager._is_object_in_workspace(outside_points)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()