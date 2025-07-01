from .base_estimator import BaseEstimator
from ...utils.icp_utils import ICPPoseEstimator  # Reuse existing utility

class ICPEstimator(BaseEstimator):
    """ICP-specific pose estimation implementation"""
    
    def __init__(self, node, icp_config: dict, shared_config: dict):
        super().__init__(node, icp_config, shared_config)
        
        # ICP-specific configuration
        self.model_folder_path = icp_config.get('model_folder_path', '')
        self.distance_threshold = icp_config.get('distance_threshold', 0.03)
        self.visualize_alignment = icp_config.get('visualize_icp_alignment', False)
        
        # Initialize ICP estimator using existing utility
        self.icp_estimator = ICPPoseEstimator(
            model_folder_path=self.model_folder_path,
            distance_threshold=self.distance_threshold,
            visualize=self.visualize_alignment,
            logger=self.logger
        )
    
    def initialize(self) -> bool:
        return self.ros_publisher.initialize()
    
    def process_objects(self, object_point_clouds, object_classes, workspace_cloud=None):
        results = []
        
        for i, (point_cloud, class_id) in enumerate(zip(object_point_clouds, object_classes)):
            result = self._process_single_object(point_cloud, class_id, i, workspace_cloud)
            if result:
                results.append(result)
        
        return results
    
    def _process_single_object(self, point_cloud, class_id, object_id, workspace_cloud):
        # Preprocess using shared point cloud processor
        processed_points = self.point_cloud_processor.preprocess(point_cloud, class_id)
        if processed_points is None:
            return None
        
        # Estimate pose using existing ICP utility
        pose_matrix = self.icp_estimator.estimate_pose(processed_points, class_id, workspace_cloud)
        
        if pose_matrix is not None:
            # Publish using shared ROS publisher
            self.ros_publisher.publish_pose(pose_matrix, class_id, object_id)
            
            return {
                'method': 'icp',
                'class_id': class_id,
                'pose_matrix': pose_matrix,
                'processed_points': processed_points,
                'object_id': object_id
            }
        
        return None
    
    def _needs_visualization(self) -> bool:
        return (self.shared_config.get('visualize_fused_point_clouds', False) or 
                self.shared_config.get('enable_detected_object_clouds_visualization', False))
    
    def is_ready(self) -> bool:
        return self.icp_estimator is not None