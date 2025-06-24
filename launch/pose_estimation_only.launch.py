from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='superquadric_grasp_system',
            executable='perception_node',
            name='perception_node',
            parameters=[{
                'pose_estimation_method': 'superquadric',  # or 'icp'
                'grasp_planning_enabled': False,  # Just pose estimation
                'detection_method': 'yolo',
                'publish_point_clouds': True,
                'web_enabled': False,
                'visualize_3d': True,
            }],
            output='screen'
        )
    ])