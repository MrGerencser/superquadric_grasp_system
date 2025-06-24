from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='superquadric_grasp_system',
            executable='perception_node',
            name='perception_node',
            parameters=[{
                'pose_estimation_method': 'icp',
                'icp_model_path': '/path/to/your/reference_model.ply',
                'icp_distance_threshold': 0.03,
                'grasp_planning_enabled': False,
                'detection_method': 'yolo',
                'publish_point_clouds': True,
            }],
            output='screen'
        )
    ])