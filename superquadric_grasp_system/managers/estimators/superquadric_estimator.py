# superquadric_grasp_system/managers/estimators/superquadric_estimator.py 
from ...utils.superquadric_utils import fit_single_superquadric, fit_multiple_superquadrics
from ...utils.superquadric_grasp_planning.superquadric_grasp_planner import SuperquadricGraspPlanner
from ...utils.point_cloud_utils import PointCloudProcessor
from ...utils.ros_publisher import ROSPublisher
from ...visualization.main_visualizer import PerceptionVisualizer

from scipy.spatial.transform import Rotation as R_simple
import numpy as np
from typing import Dict


class SuperquadricEstimator:
    def __init__(self, node, outlier_ratio: float, use_kmeans_clustering: bool,
                gripper_jaw_length: float, gripper_max_opening: float,
                class_names: Dict[int, str], poisson_reconstruction: bool,
                outlier_removal: bool, voxel_downsample_size: float,
                enable_superquadric_fit_visualization: bool = False,
                enable_all_valid_grasps_visualization: bool = False,
                enable_best_grasps_visualization: bool = False,
                enable_support_test_visualization: bool = False,
                enable_collision_test_visualization: bool = False,
                grasp_planning_enabled: bool = True):
        self.node = node
        self.logger = node.get_logger()
        
        # Direct assignment - no config lookup needed
        self.outlier_ratio = outlier_ratio
        self.use_kmeans_clustering = use_kmeans_clustering
        self.gripper_jaw_length = gripper_jaw_length
        self.gripper_max_opening = gripper_max_opening
        self.class_names = class_names
        self.enable_superquadric_fit_visualization = enable_superquadric_fit_visualization

        # Visualization flags
        self.enable_all_valid_grasps_visualization = enable_all_valid_grasps_visualization
        self.enable_best_grasps_visualization = enable_best_grasps_visualization
        self.enable_support_test_visualization = enable_support_test_visualization
        self.enable_collision_test_visualization = enable_collision_test_visualization
        
        # Grasp planning flag
        self.grasp_planning_enabled = grasp_planning_enabled
        
        # Initialize visualizer if needed
        self.visualizer = None
        if self._needs_visualization():
            try:
                self.visualizer = PerceptionVisualizer()
            except ImportError as e:
                self.logger.warning(f"Could not import visualizer: {e}")
                self.visualizer = None
        
        # Create point cloud processor with explicit config
        self.point_cloud_processor = PointCloudProcessor(
            poisson_reconstruction=poisson_reconstruction,
            outlier_removal=outlier_removal,
            voxel_downsample_size=voxel_downsample_size
        )
        
        # Create ROS publisher with explicit config
        self.ros_publisher = ROSPublisher(
            node,
            {
                'class_names': class_names,
                'publish_poses': True,
                'publish_transforms': True,
                'target_frame': 'panda_link0'
            }
        )
        
        # Initialize grasp planner
        self.grasp_planner = SuperquadricGraspPlanner(
            jaw_len=self.gripper_jaw_length,
            max_open=self.gripper_max_opening
        )
    
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
        if self.enable_superquadric_fit_visualization and self.visualizer:
            self.visualizer.visualize_superquadric_fit(processed_points, all_sqs, class_id)
        
        # Generate grasp poses using integrated planner
        all_grasp_poses, best_grasp_pose, best_sq_index = self._generate_grasp_poses(processed_points, all_sqs, class_id)
        
        # Visualize grasps if enabled
        self._visualize_grasps(processed_points, all_sqs, all_grasp_poses, best_grasp_pose, class_id, best_sq_index)
        
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
            return [], None, None
            
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
            all_grasp_data = []
            
            # Generate grasps for each superquadric
            for sq_ind, sq in enumerate(all_sqs):
                try:
                    sq_params = self._extract_sq_params(sq)
                    
                    # Use plan_filtered_grasps which includes proper filtering
                    filtered_grasps_data = self.grasp_planner.plan_filtered_grasps(
                        point_cloud_path=temp_path,
                        shape=sq_params['shape'],
                        scale=sq_params['scale'],
                        euler=sq_params['euler'],
                        translation=sq_params['translation'],
                        max_grasps=5,
                        visualize_support_test=self.enable_support_test_visualization,
                        visualize_collision_test=self.enable_collision_test_visualization
                    )
                    
                    # Only use the filtered results from plan_filtered_grasps
                    if filtered_grasps_data:
                        for grasp_data in filtered_grasps_data:
                            grasp_data['sq_index'] = sq_ind  # Track which SQ this came from

                        all_grasp_data.extend(filtered_grasps_data)

                        # Extract poses for visualization
                        for grasp_data in filtered_grasps_data:
                            if isinstance(grasp_data, dict) and 'pose' in grasp_data:
                                all_grasp_poses.append(grasp_data['pose'])
                                
                except Exception as sq_error:
                    self.logger.warning(f"Failed to generate grasps for superquadric: {sq_error}")
                    continue
            
            # Select best grasp from the FILTERED results only
            best_grasp_pose = None
            best_sq_index = None
            if all_grasp_data:
                try:
                    # Use the sophisticated selection criteria instead of simple max score
                    best_grasp_data = self.grasp_planner.select_best_grasp_with_criteria(
                        grasp_data_list=all_grasp_data,
                        point_cloud=processed_points
                    )
                    
                    if best_grasp_data and isinstance(best_grasp_data, dict) and 'pose' in best_grasp_data:
                        best_grasp_pose = best_grasp_data['pose']
                        best_sq_index = best_grasp_data.get('sq_index', 0)  # Get the SQ index
                        self.logger.info(f"Selected best grasp from SQ {best_sq_index + 1} with score: {best_grasp_data.get('score', 'N/A')}")
            
                except Exception as selection_error:
                    self.logger.warning(f"Sophisticated grasp selection failed: {selection_error}")
                    # Fallback to simple selection
                    best_grasp_data = max(all_grasp_data, key=lambda x: x.get('score', 0.0))
                    if best_grasp_data and 'pose' in best_grasp_data:
                        best_grasp_pose = best_grasp_data['pose']
                        best_sq_index = best_grasp_data.get('sq_index', 0)  # Get the SQ index

            # Clean up temporary file
            try:
                import os
                os.unlink(temp_path)
            except:
                pass

            return all_grasp_poses, best_grasp_pose, best_sq_index

        except Exception as e:
            self.logger.error(f"Grasp pose generation failed: {e}")
            return [], None, None

    def _visualize_grasps(self, processed_points, all_sqs, all_grasp_poses, best_grasp_pose, class_id, best_sq_index=None):
        """Handle all superquadric grasp visualizations"""
        if not self.visualizer:
            return
            
        try:
            # Convert ALL superquadric objects to visualization format (for all grasps view)
            all_superquadric_params = []
            for sq in all_sqs:
                sq_params = self._extract_sq_params(sq)
                all_superquadric_params.append(sq_params)
            
            # All valid grasps visualization (show all superquadrics)
            if self.enable_all_valid_grasps_visualization and all_grasp_poses:
                try:
                    self.visualizer.visualize_grasps(
                        point_cloud_data=processed_points,
                        superquadric_params=all_superquadric_params,  # All SQs
                        grasp_poses=all_grasp_poses,
                        gripper_colors=[(1, 0, 0)] * len(all_grasp_poses),
                        show_sweep_volume=False,
                        window_name=f"All Grasps: {self.class_names.get(class_id, f'Object_{class_id}')}"
                    )
                except Exception as viz_error:
                    self.logger.error(f"All valid grasp visualization failed: {viz_error}")
            
            # Best grasp visualization (show ONLY the best SQ)
            if self.enable_best_grasps_visualization and best_grasp_pose is not None:
                try:
                    # Get only the superquadric associated with the best grasp
                    if best_sq_index is not None and 0 <= best_sq_index < len(all_sqs):
                        best_sq = all_sqs[best_sq_index]
                        best_superquadric_params = [self._extract_sq_params(best_sq)]
                        sq_info = f"SQ {best_sq_index + 1}"
                    else:
                        # Fallback: use all superquadrics
                        best_superquadric_params = all_superquadric_params
                        sq_info = "All SQs"
                    
                    self.visualizer.visualize_grasps(
                        point_cloud_data=processed_points,
                        superquadric_params=best_superquadric_params,  # Only best SQ
                        grasp_poses=[best_grasp_pose],
                        gripper_colors=[(0, 1, 0)],
                        show_sweep_volume=True,
                        window_name=f"Best Grasp ({sq_info}): {self.class_names.get(class_id, f'Object_{class_id}')}"
                    )
                    self.logger.info(f"Best grasp visualization completed for {sq_info}")
                except Exception as viz_error:
                    self.logger.error(f"Best grasp visualization failed: {viz_error}")
                    
        except Exception as e:
            self.logger.error(f"Error in superquadric grasp visualization: {e}")
    
    def _needs_visualization(self) -> bool:
        return any([
            self.enable_superquadric_fit_visualization,
            self.enable_all_valid_grasps_visualization, 
            self.enable_best_grasps_visualization,
            self.enable_support_test_visualization,
            self.enable_collision_test_visualization,
        ])
        
    def is_ready(self) -> bool:
        """Check if superquadric estimator is ready"""
        if not self.grasp_planning_enabled:
            self.logger.warning("Grasp planning is disabled, skipping readiness check for planner.")
            return False
        
        # Check if ROS publisher is ready
        if not self.ros_publisher or not hasattr(self.ros_publisher, 'pose_publisher'):
            self.logger.error("ROS publisher is not initialized or missing pose publisher.")
            return False
        
        # Check if grasp planner is ready (if grasp planning is enabled)
        if self.grasp_planning_enabled and not self.grasp_planner:
            self.logger.error("Grasp planner is not initialized.")
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