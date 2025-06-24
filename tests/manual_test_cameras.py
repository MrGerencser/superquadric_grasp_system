#!/usr/bin/env python3
"""
Manual test script for testing camera manager with real hardware
"""
import rclpy
from rclpy.node import Node
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from superquadric_grasp_system.managers.camera_manager import CameraManager

def test_cameras():
    """Test camera manager with real ZED cameras"""
    
    # Initialize ROS
    rclpy.init()
    node = Node('test_cameras_manual')
    
    try:
        # Configuration for cameras
        config = {
            'camera1_sn': 33137761,  # Replace with your camera serial numbers
            'camera2_sn': 36829049,
            'device': 'cuda',
            'transform_file_path': 'launch/config/transform.yaml'  # Update this path
        }
        
        # Create camera manager
        print("Creating camera manager...")
        manager = CameraManager(node, config)
        
        # Initialize
        print("Initializing cameras (this may take a moment)...")
        if not manager.initialize():
            print("Failed to initialize cameras - check hardware connections")
            return
        
        print("Cameras initialized successfully!")
        print(f"Camera 1 ready: {manager.zed1 is not None}")
        print(f"Camera 2 ready: {manager.zed2 is not None}")
        print(f"Manager ready: {manager.is_ready()}")
        
        # Test frame capture
        print("\nTesting frame capture...")
        for i in range(5):
            print(f"Capturing frame {i+1}/5...")
            
            frame1, frame2 = manager.capture_frames()
            
            if frame1 is not None and frame2 is not None:
                print(f"  Success! Frame shapes: {frame1.shape}, {frame2.shape}")
            else:
                print(f"  Failed to capture frames")
            
            time.sleep(1)
        
        # Test depth maps
        print("\nTesting depth map retrieval...")
        depth1, depth2 = manager.get_depth_maps()
        
        if depth1 is not None and depth2 is not None:
            print(f"  Depth maps retrieved! Shapes: {depth1.shape}, {depth2.shape}")
        else:
            print(f"  Failed to retrieve depth maps")
        
        # Test point clouds
        print("\nTesting point cloud retrieval...")
        pc1, pc2 = manager.get_point_clouds()
        
        if pc1 is not None and pc2 is not None:
            print(f"  Point clouds retrieved! Shapes: {pc1.shape}, {pc2.shape}")
        else:
            print(f"  Failed to retrieve point clouds")
        
        print("\nCamera test completed successfully!")
        
    except Exception as e:
        print(f"Camera test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("Cleaning up...")
        manager.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    test_cameras()