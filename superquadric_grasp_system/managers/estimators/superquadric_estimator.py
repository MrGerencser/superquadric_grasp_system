# superquadric_grasp_system/managers/estimators/superquadric_estimator.py 
from .base_estimator import BaseEstimator
from ...utils.superquadric_utils import fit_single_superquadric, fit_multiple_superquadrics
from ...utils.grasp_planning.superquadric_grasp_planner import SuperquadricGraspPlanner
from scipy.spatial.transform import Rotation as R_simple
import numpy as np

class SuperquadricEstimator(BaseEstimator):
    """Superquadric-specific pose estimation implementation"""
    
    def __init__(self, node, superquadric_config: dict, shared_config: dict):
        # Visualization flags - Match the config file keys exactly
        self.enable_fit_visualization = superquadric_config.get('enable_superquadric_fit_visualization', False)
        self.enable_all_grasps_visualization = superquadric_config.get('enable_all_valid_grasps_visualization', False)
        self.enable_best_grasp_visualization = superquadric_config.get('enable_best_grasps_visualization', False)
        self.enable_support_test_visualization = superquadric_config.get('enable_support_test_visualization', False)
        self.enable_collision_test_visualization = superquadric_config.get('enable_collision_test_visualization', False)

        super().__init__(node, superquadric_config, shared_config)
        
        # Superquadric-specific configuration
        self.enabled = superquadric_config.get('enabled', True)
        self.outlier_ratio = superquadric_config.get('outlier_ratio', 0.9)
        self.use_kmeans_clustering = superquadric_config.get('use_kmeans_clustering', False)
        self.grasp_planning_enabled = superquadric_config.get('grasp_planning_enabled', True)
        
        # Class names for visualization
        self.class_names = {0: 'Cone', 1: 'Cup', 2: 'Mallet', 3: 'Screw Driver', 4: 'Sunscreen'}
        
        # Debug logging to verify configuration
        self.logger.debug(f"SuperquadricEstimator config keys: {list(superquadric_config.keys())}")
        self.logger.debug(f"Fit visualization enabled: {self.enable_fit_visualization}")
        self.logger.debug(f"All grasps visualization enabled: {self.enable_all_grasps_visualization}")
        self.logger.debug(f"Best grasp visualization enabled: {self.enable_best_grasp_visualization}")
        
        # Initialize grasp planner directly (instead of through manager)
        if self.enabled and self.grasp_planning_enabled:
            # Extract gripper parameters from config
            self.gripper_jaw_len = superquadric_config.get('gripper_jaw_length', 0.041)
            self.gripper_max_open = superquadric_config.get('gripper_max_opening', 0.08)
            
            # Create planner directly
            self.grasp_planner = SuperquadricGraspPlanner(
                jaw_len=self.gripper_jaw_len, 
                max_open=self.gripper_max_open
            )
            
            self.logger.info(f"Initialized grasp planner with jaw_len={self.gripper_jaw_len}, max_open={self.gripper_max_open}")
            self.logger.info(f"Debug visualization - Support: {self.enable_support_test_visualization}, Collision: {self.enable_collision_test_visualization}")
        else:
            self.grasp_planner = None
    
    def initialize(self) -> bool:
        """Initialize superquadric estimator"""
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
        
        # Fit superquadrics using existing utility functions
        best_sq, all_sqs = self._fit_superquadrics(processed_points)
        if not all_sqs:
            return None
        
        # Visualize superquadric fit if enabled
        if self.enable_fit_visualization and self.visualizer:
            self.visualizer.visualize_superquadric_fit(processed_points, all_sqs, class_id)
        
        # Generate grasp poses using integrated planner
        all_grasp_poses, best_grasp_pose = self._generate_grasp_poses(processed_points, all_sqs, class_id)
        
        # Visualize grasps if enabled
        self._visualize_grasps(processed_points, all_sqs, all_grasp_poses, best_grasp_pose, class_id)
        
        # Publish best grasp pose
        if best_grasp_pose is not None:
            self.ros_publisher.publish_pose(best_grasp_pose, class_id, object_id)
        
        return {
            'method': 'superquadric',
            'class_id': class_id,
            'processed_points': processed_points,
            'superquadrics': all_sqs,
            'grasp_poses': all_grasp_poses,
            'best_grasp_pose': best_grasp_pose,
            'object_id': object_id
        }
    
    def _fit_superquadrics(self, points):
        """Use existing superquadric fitting utilities"""
        if self.use_kmeans_clustering:
            return fit_multiple_superquadrics(points, self.outlier_ratio, self.logger)
        else:
            return fit_single_superquadric(points, self.outlier_ratio, self.logger)
    
    def _generate_grasp_poses(self, processed_points, all_sqs, class_id):
        """Generate grasp poses using integrated grasp planner directly"""
        if not self.grasp_planner or not all_sqs:
            return [], None
            
        try:
            # Save point cloud to temporary file for grasp planning
            import tempfile
            import open3d as o3d
            
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
                temp_path = tmp_file.name
                
            # Save points as PLY file
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(processed_points)
            o3d.io.write_point_cloud(temp_path, pcd)
            
            all_grasp_poses = []
            all_grasp_data = []  # Store all grasp data for best selection
            
            # Generate grasps for each superquadric
            for sq in all_sqs:
                try:
                    sq_params = self._extract_sq_params(sq)
                    
                    # Plan grasps directly using SuperquadricGraspPlanner
                    grasps_data = self.grasp_planner.plan_grasps(
                        point_cloud_path=temp_path,
                        shape=sq_params['shape'],
                        scale=sq_params['scale'],
                        euler=sq_params['euler'],
                        translation=sq_params['translation'],
                        max_grasps=5,
                        visualize_support_test=self.enable_support_test_visualization,
                        visualize_collision_test=self.enable_collision_test_visualization
                    )
                    
                    # Collect all grasp data for combined selection
                    if grasps_data:
                        all_grasp_data.extend(grasps_data)
                        # Extract poses for visualization
                        for grasp_data in grasps_data:
                            if isinstance(grasp_data, dict) and 'pose' in grasp_data:
                                all_grasp_poses.append(grasp_data['pose'])
                            
                except Exception as sq_error:
                    self.logger.warning(f"Failed to generate grasps for superquadric: {sq_error}")
                    continue
            
            # Now select the best grasp from all collected grasps using the proper criteria
            best_grasp_pose = None
            if all_grasp_data:
                try:
                    # Calculate object center from all superquadrics
                    object_center = None
                    if all_sqs:
                        centers = []
                        for sq in all_sqs:
                            if hasattr(sq, 'translation'):
                                centers.append(sq.translation)
                            elif isinstance(sq, dict) and 'translation' in sq:
                                centers.append(sq['translation'])
                        if centers:
                            object_center = np.mean(centers, axis=0)
                    
                    # Use the planner's best selection method directly
                    best_grasp_data = self.grasp_planner.select_best_grasp_with_criteria(
                        all_grasp_data, 
                        object_center=object_center,
                        point_cloud=processed_points
                    )
                    
                    if best_grasp_data and isinstance(best_grasp_data, dict) and 'pose' in best_grasp_data:
                        best_grasp_pose = best_grasp_data['pose']
                        self.logger.info(f"Selected best grasp with score: {best_grasp_data.get('score', 'N/A')}")
                    
                except Exception as selection_error:
                    self.logger.warning(f"Best grasp selection failed, using fallback: {selection_error}")
                    # Fallback: use first grasp if available
                    if all_grasp_data:
                        first_grasp = all_grasp_data[0]
                        if isinstance(first_grasp, dict) and 'pose' in first_grasp:
                            best_grasp_pose = first_grasp['pose']
            
            # Clean up temporary file
            try:
                import os
                os.unlink(temp_path)
            except:
                pass
                
            return all_grasp_poses, best_grasp_pose
            
        except Exception as e:
            self.logger.error(f"Grasp pose generation failed: {e}")
            return [], None
    
    def _visualize_grasps(self, processed_points, all_sqs, all_grasp_poses, best_grasp_pose, class_id):
        """Handle all superquadric grasp visualizations"""
        if not self.visualizer:
            return
            
        try:
            # Convert superquadric objects to visualization format
            superquadric_params = []
            for sq in all_sqs:
                sq_params = self._extract_sq_params(sq)
                superquadric_params.append(sq_params)
            
            # All valid grasps visualization
            if self.enable_all_grasps_visualization and all_grasp_poses:
                try:
                    self.visualizer.visualize_grasps(
                        point_cloud_data=processed_points,
                        superquadric_params=superquadric_params,
                        grasp_poses=all_grasp_poses,
                        gripper_colors=[(1, 0, 0)] * len(all_grasp_poses),  # Red color for all grasps
                        show_sweep_volume=False,
                        window_name=f"All Grasps: {self.class_names.get(class_id, f'Object_{class_id}')}"
                    )
                except Exception as viz_error:
                    self.logger.error(f"All valid grasp visualization failed: {viz_error}")
            
            # Best grasp visualization
            if self.enable_best_grasp_visualization and best_grasp_pose is not None:
                try:
                    self.visualizer.visualize_grasps(
                        point_cloud_data=processed_points,
                        superquadric_params=superquadric_params,
                        grasp_poses=[best_grasp_pose],  # Only the best grasp
                        gripper_colors=[(0, 1, 0)],  # Green color for best grasp
                        show_sweep_volume=True,
                        window_name=f"Best Grasp: {self.class_names.get(class_id, f'Object_{class_id}')}"
                    )
                    self.logger.info(f"Best grasp visualization completed")
                except Exception as viz_error:
                    self.logger.error(f"Best grasp visualization failed: {viz_error}")
                    
        except Exception as e:
            self.logger.error(f"Error in superquadric grasp visualization: {e}")
    
    def _needs_visualization(self) -> bool:
        return any([
            self.enable_fit_visualization,
            self.enable_all_grasps_visualization,
            self.enable_best_grasp_visualization,
            self.shared_config.get('visualize_fused_point_clouds', False),
            self.shared_config.get('enable_detected_object_clouds_visualization', False)
        ])
        
    def is_ready(self) -> bool:
        """Check if superquadric estimator is ready"""
        if not self.enabled:
            return False
        
        # Check if ROS publisher is ready
        if not self.ros_publisher or not hasattr(self.ros_publisher, 'pose_publisher'):
            return False
        
        # Check if grasp planner is ready (if grasp planning is enabled)
        if self.grasp_planning_enabled and not self.grasp_planner:
            return False
        
        return True
    
    def _extract_sq_params(self, sq):
        """Extract superquadric parameters in a unified way"""
        if hasattr(sq, 'shape'):
            # Handle superquadric object
            euler_angles = (R_simple.from_matrix(sq.RotM).as_euler('xyz') 
                        if hasattr(sq, 'RotM') else np.array([0, 0, 0]))
            
            return {
                'shape': np.asarray(sq.shape) if hasattr(sq.shape, '__len__') else np.array([sq.shape, sq.shape]),
                'scale': np.asarray(sq.scale) if hasattr(sq.scale, '__len__') else np.array([sq.scale, sq.scale, sq.scale]),
                'euler': euler_angles,
                'translation': np.asarray(sq.translation) if hasattr(sq, 'translation') else np.array([0, 0, 0])
            }
        else:
            # Handle dict format
            return sq