from .base_estimator import BaseEstimator
from ...utils.icp_utils import ICPPoseEstimator  # Reuse existing utility
import numpy as np
import open3d as o3d

class ICPEstimator(BaseEstimator):
    """ICP-specific pose estimation implementation"""
    
    def __init__(self, node, icp_config: dict, shared_config: dict):
        super().__init__(node, icp_config, shared_config)
        
        # ICP-specific configuration
        self.model_folder_path = icp_config.get('model_folder_path', '')
        self.distance_threshold = icp_config.get('distance_threshold', 0.03)
        self.visualize_alignment = icp_config.get('visualize_icp_alignment', False)
        self.visualize_grasps = icp_config.get('visualize_grasp_poses', False)
        self.max_grasp_candidates = icp_config.get('max_grasp_candidates', 3)
        
        self.class_names = shared_config.get('class_names', {})
        
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
            # Store the pose matrix and transformed reference model for visualization
            self._last_pose_matrix = pose_matrix
            
            # Get the transformed reference model from ICP
            if hasattr(self.icp_estimator, 'reference_models') and class_id in self.icp_estimator.reference_models:
                reference_model = self.icp_estimator.reference_models[class_id]
                # Create a copy and transform it
                self._transformed_reference_model = o3d.geometry.PointCloud(reference_model)
                self._transformed_reference_model.transform(pose_matrix)
            else:
                self._transformed_reference_model = None
            
            # Get transformed grasp poses
            grasp_poses = self.icp_estimator.get_transformed_grasp_poses(pose_matrix, class_id, max_grasps=3)
            
            # Select best grasp pose
            best_grasp_pose = self._select_best_grasp(grasp_poses, processed_points)
            
            # Visualize if enabled
            if self.visualize_grasps and self.visualizer:
                self._visualize_icp_grasps(processed_points, grasp_poses, best_grasp_pose, class_id)
            
            # Publish best grasp pose if available
            if best_grasp_pose is not None:
                self.ros_publisher.publish_pose(best_grasp_pose['pose'], class_id, object_id)
            
            return {
                'method': 'icp',
                'class_id': class_id,
                'pose_matrix': pose_matrix,
                'processed_points': processed_points,
                'grasp_poses': [g['pose'] for g in grasp_poses],
                'best_grasp_pose': best_grasp_pose['pose'] if best_grasp_pose else None,
                'object_id': object_id
            }
        
        return None
    
    def _select_best_grasp(self, grasp_poses, point_cloud):
        """Select the best grasp pose from candidates"""
        if not grasp_poses:
            return None
        
        # Simple selection criteria - you can make this more sophisticated
        best_grasp = None
        best_score = -1
        
        robot_base = np.array([0.0, 0.0, 0.0])
        
        for grasp in grasp_poses:
            try:
                position = np.array(grasp['position'])
                
                # Distance to robot base (prefer closer grasps)
                distance_to_base = np.linalg.norm(position - robot_base)
                distance_score = 1.0 / (1.0 + distance_to_base)
                
                # Height preference (prefer higher grasps)
                height_score = 1.0 if position[2] > 0.05 else 0.5
                
                # Workspace preference
                workspace_score = 1.0 if (0.2 <= position[0] <= 0.6 and 
                                        -0.3 <= position[1] <= 0.3) else 0.5
                
                # Combined score
                final_score = (distance_score * 1.5 + 
                             height_score * 1.0 + 
                             workspace_score * 1.0)
                
                if final_score > best_score:
                    best_score = final_score
                    best_grasp = grasp
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error scoring grasp {grasp['name']}: {e}")
                continue
        
        if self.logger and best_grasp:
            self.logger.info(f"Selected best grasp: {best_grasp['name']} with score: {best_score:.3f}")
        
        return best_grasp
    
    def _visualize_icp_grasps(self, processed_points, grasp_poses, best_grasp_pose, class_id):
        """Visualize ICP-based grasp poses"""
        if not grasp_poses:
            return
            
        try:
            all_poses = [g['pose'] for g in grasp_poses]
            best_pose = best_grasp_pose['pose'] if best_grasp_pose else None
            
            # Convert point cloud to numpy array
            if hasattr(processed_points, 'points'):
                point_cloud_array = np.asarray(processed_points.points)
            else:
                point_cloud_array = processed_points
            
            class_name = self.class_names.get(class_id, f'Object_{class_id}')
            
            # Get the transformed reference model
            transformed_reference_model = getattr(self, '_transformed_reference_model', None)
            
            # Use existing visualization system with reference model
            self.visualizer.visualize_grasps(
                point_cloud_data=point_cloud_array,
                superquadric_params=None,  # No superquadric for ICP
                grasp_poses=all_poses,
                gripper_colors=[(0, 0, 1)] * len(all_poses),  # Blue for all grasps
                show_sweep_volume=False,
                reference_model=transformed_reference_model,  # Use transformed reference model
                window_name=f"ICP Grasps: {class_name}"
            )
            
            # Show best grasp separately
            if best_pose is not None:
                self.visualizer.visualize_grasps(
                    point_cloud_data=point_cloud_array,
                    superquadric_params=None,
                    grasp_poses=[best_pose],
                    gripper_colors=[(0, 1, 0)],  # Green for best grasp
                    show_sweep_volume=True,
                    reference_model=transformed_reference_model,
                    window_name=f"Best ICP Grasp: {class_name}"
                )
                
        except Exception as e:
            self.logger.error(f"ICP grasp visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _needs_visualization(self) -> bool:
        return (self.shared_config.get('visualize_fused_point_clouds', False) or 
                self.shared_config.get('enable_detected_object_clouds_visualization', False))
    
    def is_ready(self) -> bool:
        return self.icp_estimator is not None