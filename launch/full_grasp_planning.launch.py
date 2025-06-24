from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='superquadric_grasp_system',
            executable='perception_node',
            name='perception_node',
            parameters=[{
                'pose_estimation_method': 'superquadric',
                'grasp_planning_enabled': True,  # Full grasp planning
                'detection_method': 'yolo',
                'publish_point_clouds': True,
                'web_enabled': True,
                'visualize_3d': True,
            }],
            output='screen'
        )
    ])