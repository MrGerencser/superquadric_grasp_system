import unittest
import numpy as np
import rclpy
from rclpy.node import Node
from superquadric_grasp_system.config.system_config import SystemConfig

class TestSystemIntegration(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('test_integration')
        
        # Mock configuration for testing
        self.mock_config = {
            'processing_rate': 1.0,  # Slow for testing
            'device': 'cpu',
            'camera': {
                'camera1_sn': 33137761,
                'camera2_sn': 36829049,
                'transform_file_path': '/dev/null'
            },
            'detection': {
                'yolo_model_path': 'superquadric_grasp_system/models/yolo_seg_finetuned.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu',
                'classes': [0],
                'class_names': {0: 'test_object'}
            },
            'point_cloud': {
                'workspace_bounds': [-0.5, 0.5, -0.5, 0.5, 0.0, 1.0],
                'voxel_size': 0.01,
                'distance_threshold': 0.1,
                'require_both_cameras': False,
                'device': 'cpu'
            },
            'pose_estimation': {
                'superquadric_enabled': False,  # Disable for basic testing
                'gripper_jaw_length': 0.04,
                'gripper_max_opening': 0.08,
                'outlier_ratio': 0.9,
                'use_kmeans_clustering': False,
                'require_both_cameras': False,
                'poisson_reconstruction': False
            },
            'visualization': {
                'web_enabled': False,  # Disable for testing
                'publish_point_clouds': True,
                'target_frame': 'base_link',
                'visualize_yolo': False,
                'visualize_3d': False
            }
        }
    
    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()
    
    def test_system_coordinator_creation(self):
        """Test that system coordinator can be created"""
        coordinator = SystemCoordinator(self.node, self.mock_config)
        self.assertIsNotNone(coordinator)
    
    def test_manager_initialization_flow(self):
        """Test the initialization flow of all managers"""
        coordinator = SystemCoordinator(self.node, self.mock_config)
        
        # Test individual manager creation
        self.assertIsNotNone(coordinator.camera_manager)
        self.assertIsNotNone(coordinator.detection_manager)
        self.assertIsNotNone(coordinator.point_cloud_manager)
        self.assertIsNotNone(coordinator.pose_estimation_manager)
        self.assertIsNotNone(coordinator.visualization_manager)
        
        # Test that at least some managers can initialize
        # (Camera manager will fail without hardware, but others should work)
        self.assertTrue(coordinator.point_cloud_manager.initialize())
        self.assertTrue(coordinator.visualization_manager.initialize())

if __name__ == '__main__':
    unittest.main()