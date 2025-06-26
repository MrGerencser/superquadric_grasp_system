import os
import numpy as np
import time
from typing import List, Tuple, Optional, Any
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header, Bool
from scipy.spatial.transform import Rotation as R_simple
from sklearn.cluster import KMeans
import open3d as o3d
import traceback
from tf2_ros import TransformBroadcaster

from .base_manager import BaseManager
from ..utils.transform_utils import create_pose_message
from ..utils.superquadric_utils import (
    fit_single_superquadric,
    fit_multiple_superquadrics,
    )
from ..utils.icp_utils import ICPPoseEstimator
from .grasp_planning_manager import GraspPlanningManager
from ..visualization.main_visualizer import GraspVisualizer


class PoseEstimationManager(BaseManager):
    """Manages pose estimation using either ICP or superquadric fitting
       Can be also extended to support other methods in the future
    """
    
    def __init__(self, node, config):
        super().__init__(node, config)
        
        # Estimation method selection
        self.method = config.get('method', 'superquadric')  # 'icp' or 'superquadric'
        
        # Extract shared configuration
        shared_config = config.get('shared', {})
        self._initialize_shared_components(shared_config)
        
        # Method-specific configuration
        if self.method == 'icp':
            icp_config = config.get('icp', {})
            self._initialize_icp_components(icp_config)
            self.superquadric_config = {}
        else:
            superquadric_config = config.get('superquadric', {})
            self._initialize_superquadric_components(superquadric_config)
            self.icp_config = {}
        
        # Class names mapping
        self.class_names = {0: 'Cone', 1: 'Cup', 2: 'Mallet', 3: 'Screw Driver', 4: 'Sunscreen'}
        self.graspable_classes = [0, 1, 2, 3, 4]
        
        # Grasp execution state tracking
        self.grasp_executing = False
        
        # ROS publishers and TF
        self.pose_publisher = None
        self.tf_broadcaster = None
        
        # Initialize visualization
        self._initialize_visualization()
    
    def _initialize_shared_components(self, shared_config):
        """Initialize shared components used by both methods"""
        # Point cloud preprocessing parameters
        self.poisson_reconstruction = shared_config.get('poisson_reconstruction', False)
        self.outlier_removal = shared_config.get('outlier_removal', True)
        self.voxel_downsample_size = shared_config.get('voxel_downsample_size', 0.002)
        
        # Shared visualization parameters
        self.visualize_fused_point_clouds = shared_config.get('visualize_fused_point_clouds', False)
        self.enable_detected_object_clouds_visualization = shared_config.get('enable_detected_object_clouds_visualization', False)
        
        # Publishing parameters
        self.publish_poses = shared_config.get('publish_poses', True)
        self.publish_transforms = shared_config.get('publish_transforms', True)
        self.target_frame = shared_config.get('target_frame', 'panda_link0')
        
        self.logger.info(f"Shared components initialized:")
        self.logger.info(f"  - Poisson reconstruction: {self.poisson_reconstruction}")
        self.logger.info(f"  - Visualize fused point clouds: {self.visualize_fused_point_clouds}")
        self.logger.info(f"  - Visualize detected object clouds: {self.enable_detected_object_clouds_visualization}")
    
    def _initialize_icp_components(self, icp_config):
        """Initialize ICP-specific components"""
        self.icp_model_folder_path = icp_config.get('model_folder_path', '')
        self.icp_distance_threshold = icp_config.get('distance_threshold', 0.03)
        self.icp_max_iterations = icp_config.get('max_iterations', 50)
        
        # ICP-specific visualization
        self.visualize_icp_alignment = icp_config.get('visualize_icp_alignment', False)
        self.visualize_reference_models = icp_config.get('visualize_reference_models', False)
        self.visualize_pca_alignment = icp_config.get('visualize_pca_alignment', False)
        
        self.icp_estimator = ICPPoseEstimator(
            model_folder_path=self.icp_model_folder_path,
            distance_threshold=self.icp_distance_threshold,
            visualize=self.visualize_icp_alignment,
            logger=self.logger
        )
        self.logger.info("ICP pose estimator initialized")
        self.logger.info(f"  - Model folder: {self.icp_model_folder_path}")
        self.logger.info(f"  - Distance threshold: {self.icp_distance_threshold}")
        self.logger.info(f"  - Visualization: {self.visualize_icp_alignment}")
    
    def _initialize_superquadric_components(self, superquadric_config):
        """Initialize superquadric-specific components"""
        self.superquadric_enabled = superquadric_config.get('enabled', True)
        
        # Fitting parameters
        self.outlier_ratio = superquadric_config.get('outlier_ratio', 0.9)
        self.use_kmeans_clustering = superquadric_config.get('use_kmeans_clustering', False)
        
        # Grasp planning parameters
        self.grasp_planning_enabled = superquadric_config.get('grasp_planning_enabled', True)
        self.gripper_jaw_length = superquadric_config.get('gripper_jaw_length', 0.037)
        self.gripper_max_opening = superquadric_config.get('gripper_max_opening', 0.08)
        
        # Superquadric-specific visualization
        self.enable_superquadric_fit_visualization = superquadric_config.get('enable_superquadric_fit_visualization', False)
        self.enable_all_valid_grasps_visualization = superquadric_config.get('enable_all_valid_grasps_visualization', False)
        self.enable_best_grasps_visualization = superquadric_config.get('enable_best_grasps_visualization', False)

        # Initialize grasp planning manager if enabled
        if self.superquadric_enabled and self.grasp_planning_enabled:
            self.grasp_planning_manager = GraspPlanningManager(
                config=superquadric_config
            )
            self.logger.info("Superquadric grasp planning manager initialized")
        else:
            self.grasp_planning_manager = None
            
        self.logger.info("Superquadric components initialized:")
        self.logger.info(f"  - Enabled: {self.superquadric_enabled}")
        self.logger.info(f"  - Grasp planning: {self.grasp_planning_enabled}")
        self.logger.info(f"  - K-means clustering: {self.use_kmeans_clustering}")
    
    def _initialize_visualization(self):
        """Initialize visualization components"""
        # Determine if any visualization is needed
        need_visualizer = False
        
        if self.method == 'superquadric':
            need_visualizer = any([
                self.enable_superquadric_fit_visualization,
                self.enable_all_valid_grasps_visualization,
                self.enable_best_grasps_visualization,
            ])
        
        # Shared visualization always needs visualizer if enabled
        need_visualizer = need_visualizer or any([
            self.visualize_fused_point_clouds,
            self.enable_detected_object_clouds_visualization
        ])
        
        if need_visualizer:
            try:
                self.visualizer = GraspVisualizer()
                self.logger.info("GraspVisualizer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GraspVisualizer: {e}")
                self.visualizer = None
        else:
            self.visualizer = None

    def initialize(self) -> bool:
        """Initialize pose estimation components"""
        try:
            # Initialize pose publisher if enabled
            if self.publish_poses:
                self.pose_publisher = self.node.create_publisher(
                    PoseStamped, '/perception/object_pose', 1
                )
            
            # Initialize TF broadcaster if enabled
            if self.publish_transforms:
                self.tf_broadcaster = TransformBroadcaster(self.node)
            
            # Subscribe to grasp execution state
            self.execution_state_subscriber = self.node.create_subscription(
                Bool, '/robot/grasp_executing',
                self._grasp_execution_callback, 1
            )
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pose estimation manager: {e}")
            return False

    def process_objects(self, object_point_clouds, object_classes, workspace_cloud=None):
        """Process detected objects for pose estimation"""
        try:
            results = []
            
            # Visualize fused workspace if enabled (shared visualization)
            if self.visualize_fused_point_clouds and workspace_cloud is not None and self.visualizer:
                try:
                    if hasattr(workspace_cloud, 'points'):
                        workspace_points = np.asarray(workspace_cloud.points)
                    elif isinstance(workspace_cloud, np.ndarray):
                        workspace_points = workspace_cloud
                    else:
                        workspace_points = None
                    
                    if workspace_points is not None and len(workspace_points) > 0:
                        self._visualize_fused_workspace(workspace_points)
                except Exception as viz_error:
                    self.logger.warning(f"Fused workspace visualization failed: {viz_error}")
            
            for i, (point_cloud, class_id) in enumerate(zip(object_point_clouds, object_classes)):
                
                if self.method == 'icp':
                    # ============= ICP POSE ESTIMATION =============
                    result = self._process_object_with_icp(point_cloud, class_id, i, workspace_cloud)
                    if result:
                        results.append(result)
                        
                elif self.method == 'superquadric':
                    # ============= SUPERQUADRIC POSE ESTIMATION =============
                    result = self._process_object_with_superquadric(point_cloud, class_id, i, workspace_cloud)
                    if result:
                        results.append(result)
                        
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pose estimation processing: {e}")
            self.logger.error(traceback.format_exc())
            return []
    
    def _visualize_fused_workspace(self, workspace_points):
        """Visualize fused workspace point cloud (shared visualization)"""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(workspace_points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for workspace
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            o3d.visualization.draw_geometries(
                [pcd, coord_frame],
                window_name="Fused Workspace Point Cloud",
                zoom=0.7,
                front=[0, -1, 0],
                lookat=pcd.get_center() if len(pcd.points) > 0 else [0, 0, 0],
                up=[0, 0, 1]
            )
            
        except Exception as e:
            self.logger.warning(f"Workspace visualization failed: {e}")

    def _preprocess_point_cloud(self, object_points: np.ndarray, class_id: int = None) -> Optional[np.ndarray]:
        """Shared point cloud preprocessing for both ICP and superquadric methods"""
        try:
            # Store original points for visualization
            original_object_points = object_points.copy()
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(object_points)
            
            # Apply outlier removal if enabled
            if self.outlier_removal:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
                pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
            
            # Downsample
            if self.voxel_downsample_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_downsample_size)
            
            if len(pcd.points) < 50:  # Minimum threshold for both methods
                self.logger.warning(f"Too few points after preprocessing: {len(pcd.points)}")
                return None
            
            filtered_object_points = np.asarray(pcd.points)
            
            # Apply Poisson reconstruction if enabled (shared)
            if self.poisson_reconstruction:
                filtered_object_points = self._apply_poisson_reconstruction(
                    pcd, filtered_object_points, original_object_points
                )
            
            # Shared visualization: detected cloud filtering
            if self.enable_detected_object_clouds_visualization and self.visualizer:
                try:
                    object_center = np.mean(original_object_points, axis=0)
                    self.visualizer.visualize_detected_cloud_filtering(
                        original_points=original_object_points,
                        filtered_points=filtered_object_points,
                        class_id=class_id if class_id is not None else 0,
                        object_center=object_center
                    )
                except Exception as viz_error:
                    self.logger.warning(f"Detected cloud filtering visualization failed: {viz_error}")
            
            return filtered_object_points
            
        except Exception as e:
            self.logger.error(f"Error preprocessing point cloud: {e}")
            return None
    
    def _apply_poisson_reconstruction(self, pcd, filtered_points, original_points):
        """Apply Poisson reconstruction (shared method)"""
        try:
            # Safety check: minimum points required
            if len(filtered_points) < 100:
                self.logger.warning("Too few points for Poisson reconstruction, using original")
                return filtered_points
                
            # Check point cloud dimensionality
            bounds = pcd.get_axis_aligned_bounding_box()
            dimensions = bounds.get_extent()
            if np.min(dimensions) < 0.001:  # Very thin in one dimension
                self.logger.warning("Point cloud too thin for Poisson reconstruction")
                return filtered_points
                
            self.logger.info("Applying Poisson reconstruction...")
            
            # Estimate normals with more robust parameters
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            
            # Check if normals were computed successfully
            if not pcd.has_normals() or len(pcd.normals) == 0:
                self.logger.warning("Failed to compute normals for Poisson reconstruction")
                return filtered_points
                
            # Orient normals more carefully
            try:
                pcd.orient_normals_consistent_tangent_plane(k=50)
            except Exception as normal_error:
                self.logger.warning(f"Normal orientation failed: {normal_error}")
                return filtered_points

            # Perform Poisson reconstruction with conservative parameters
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, 
                depth=8,  # Reduced from 9
                width=0, 
                scale=1.0,  # Reduced from 1.1
                linear_fit=False
            )
            
            # Validate mesh
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                self.logger.warning("Poisson reconstruction produced empty mesh")
                return filtered_points
                
            # Remove low-density vertices (noise reduction)
            densities = np.asarray(densities)
            if len(densities) == 0:
                self.logger.warning("No density information from Poisson reconstruction")
                return filtered_points
                
            density_threshold = np.quantile(densities, 0.2)  # More conservative
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Sample points from the reconstructed mesh
            num_points = min(max(len(filtered_points), 1000), 5000)  # Cap at 5000
            try:
                reconstructed_pcd = mesh.sample_points_uniformly(number_of_points=num_points)
            except Exception as sample_error:
                self.logger.warning(f"Mesh sampling failed: {sample_error}")
                return filtered_points
                
            # Get the reconstructed points
            reconstructed_points = np.asarray(reconstructed_pcd.points)
            
            # Validate reconstruction quality
            if len(reconstructed_points) < 50:
                self.logger.warning("Poisson reconstruction produced too few points")
                return filtered_points
                
            # Check bounds reasonableness
            original_bounds = np.array([filtered_points.min(axis=0), filtered_points.max(axis=0)])
            reconstructed_bounds = np.array([reconstructed_points.min(axis=0), reconstructed_points.max(axis=0)])
            
            scale_factor = np.max((reconstructed_bounds[1] - reconstructed_bounds[0]) / 
                                (original_bounds[1] - original_bounds[0]))
            
            if scale_factor < 3.0:  # More lenient
                self.logger.info(f"Poisson reconstruction successful: {len(reconstructed_points)} points")
                return reconstructed_points
            else:
                self.logger.warning(f"Poisson reconstruction scale factor too large ({scale_factor:.2f})")
                
        except Exception as poisson_error:
            self.logger.warning(f"Poisson reconstruction failed: {poisson_error}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        # Always fall back to filtered points
        return filtered_points

    def _process_object_with_icp(self, point_cloud, class_id, object_id, workspace_cloud):
        """Process object using ICP pose estimation"""
        try:
            if class_id not in self.icp_estimator.reference_models:
                self.logger.warning(f"No reference model for class {class_id}")
                return None
            
            # Convert point cloud data to numpy array if it's not already
            if hasattr(point_cloud, 'points'):
                point_cloud_np = np.asarray(point_cloud.points)
            elif isinstance(point_cloud, np.ndarray):
                point_cloud_np = point_cloud
            else:
                self.logger.error(f"Unknown point cloud type: {type(point_cloud)}")
                return None
            
            # Shared preprocessing
            processed_points = self._preprocess_point_cloud(point_cloud_np, class_id)
            if processed_points is None or len(processed_points) < 10:
                self.logger.warning(f"ICP preprocessing failed or too few points")
                return None
            
            # Convert workspace cloud if provided
            workspace_cloud_np = None
            if workspace_cloud is not None:
                if hasattr(workspace_cloud, 'points'):
                    workspace_cloud_np = np.asarray(workspace_cloud.points)
                elif isinstance(workspace_cloud, np.ndarray):
                    workspace_cloud_np = workspace_cloud
            
            # Estimate pose using ICP
            pose_matrix = self.icp_estimator.estimate_pose(
                processed_points, class_id, workspace_cloud_np
            )
            
            if pose_matrix is not None:
                # Publish pose and transform if enabled
                if self.publish_poses and self.pose_publisher:
                    pose_msg = self.icp_estimator.matrix_to_pose_stamped(
                        pose_matrix, class_id, object_id, self.target_frame
                    )
                    pose_msg.header.stamp = self.node.get_clock().now().to_msg()
                    self.pose_publisher.publish(pose_msg)
                
                if self.publish_transforms and self.tf_broadcaster:
                    transform_msg = self.icp_estimator.create_transform_stamped(
                        pose_matrix, class_id, object_id, self.target_frame
                    )
                    transform_msg.header.stamp = self.node.get_clock().now().to_msg()
                    self.tf_broadcaster.sendTransform(transform_msg)
                
                self.logger.info(f"Published ICP pose for class {class_id}, object {object_id}")
                
                return {
                    'method': 'icp',
                    'class_id': class_id,
                    'pose_matrix': pose_matrix,
                    'processed_points': processed_points,
                    'object_id': object_id
                }
            else:
                self.logger.warning(f"ICP pose estimation failed for class {class_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in ICP processing for class {class_id}: {e}")
            return None
    
    def _process_object_with_superquadric(self, point_cloud, class_id, object_id, workspace_cloud):
        """Process object using superquadric pose estimation"""
        try:
            if not self.superquadric_enabled:
                self.logger.warning("Superquadric processing disabled")
                return None
            
            # Convert point cloud to numpy array if needed
            if hasattr(point_cloud, 'points'):
                point_cloud_np = np.asarray(point_cloud.points)
            elif isinstance(point_cloud, np.ndarray):
                point_cloud_np = point_cloud
            else:
                self.logger.error(f"Unknown point cloud type: {type(point_cloud)}")
                return None
            
            # Shared preprocessing
            processed_points = self._preprocess_point_cloud(point_cloud_np, class_id)
            if processed_points is None or len(processed_points) < 100:
                self.logger.warning(f"Superquadric preprocessing failed or too few points")
                return None
            
            self.logger.debug(f"Preprocessed {len(processed_points)} points")
            
            # Fit superquadrics
            self.logger.info("Starting superquadric fitting...")
            try:
                best_sq, all_sqs = self._fit_superquadrics(processed_points)
            except Exception as fitting_error:
                self.logger.error(f"Superquadric fitting failed: {fitting_error}")
                traceback.print_exc()
                return None
            
            if not all_sqs:
                self.logger.warning("Superquadric fitting failed")
                return None
                
            self.logger.info(f"Fitted {len(all_sqs)} superquadrics")
            
            # Visualize superquadric fit if enabled
            if self.enable_superquadric_fit_visualization and self.visualizer:
                try:
                    successful_meshes = self.visualizer.visualize_superquadric_fit(
                        points=processed_points,
                        all_sqs=all_sqs,
                        class_id=class_id,
                        class_names=self.class_names
                    )
                    self.logger.info(f"Superquadric fit visualization: {successful_meshes}/{len(all_sqs)} meshes")
                except Exception as viz_error:
                    self.logger.error(f"Superquadric fit visualization failed: {viz_error}")
            
            # Generate grasp poses
            all_grasp_poses, best_grasp_pose = self._generate_grasp_poses_from_superquadrics(
                processed_points, all_sqs, class_id
            )
            
            if not all_grasp_poses:
                self.logger.warning("No valid grasp poses generated")
                return None
                
            self.logger.info(f"Generated {len(all_grasp_poses)} grasp poses")
            
            # Publish best grasp pose 
            if self.publish_poses and best_grasp_pose is not None:
                self._publish_best_grasp_pose(best_grasp_pose, class_id, object_id)
            
            # Visualization of grasps
            self._visualize_superquadric_grasps(processed_points, all_sqs, all_grasp_poses, 
                                              best_grasp_pose, class_id)
            
            return {
                'method': 'superquadric',
                'class_id': class_id,
                'processed_points': processed_points,
                'superquadrics': all_sqs,
                'grasp_poses': all_grasp_poses,
                'best_grasp_pose': best_grasp_pose,
                'object_id': object_id
            }
            
        except Exception as e:
            self.logger.error(f"Error in superquadric processing for class {class_id}: {e}")
            traceback.print_exc()
            return None

    # ============= SUPERQUADRIC METHODS =============
    
    def _fit_superquadrics(self, points: np.ndarray) -> Tuple[Any, List[Any]]:
        """Fit superquadrics to point cloud using multiple approaches"""
        try:
            # Validate input
            if len(points) < 100:
                self.logger.warning(f"Too few points ({len(points)}) for superquadric fitting")
                return None, []
            
            # Check for recursion depth
            import sys
            current_recursion_limit = sys.getrecursionlimit()
            if current_recursion_limit < 2000:
                sys.setrecursionlimit(2000)
                self.logger.info(f"Increased recursion limit from {current_recursion_limit} to 2000")
            
            # Center points for numerical stability
            points_center = np.mean(points, axis=0)
            points_std = np.std(points, axis=0)
            
            # Check for degenerate cases
            if np.any(points_std < 1e-6):
                self.logger.warning("Degenerate point cloud detected")
                return None, []
            
            self.logger.info(f"Point cloud stats: center={points_center}, std={points_std}")
            
            if not self.use_kmeans_clustering:
                # Single superquadric approach
                self.logger.info("Using single superquadric fitting")
                return fit_single_superquadric(points, self.outlier_ratio, self.logger)
            else:
                # Multiple superquadrics approach
                self.logger.info("Using multiple superquadric ensemble fitting")
                return fit_multiple_superquadrics(points, self.outlier_ratio, self.logger)
                    
        except RecursionError as e:
            self.logger.error(f"Recursion error in superquadric fitting: {e}")
            return None, []
        except Exception as e:
            self.logger.error(f"Error fitting superquadrics: {e}")
            traceback.print_exc()
            return None, []
    
    def _generate_grasp_poses_from_superquadrics(self, processed_points: np.ndarray, 
                                            all_sqs: List, class_id: int) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Generate grasp poses using fitted superquadrics"""
        try:
            if not self.grasp_planning_manager:
                return [], None
            
            # Save temporary point cloud file for grasp planning
            import tempfile
            temp_file = tempfile.mktemp(suffix='.ply')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(processed_points)
            o3d.io.write_point_cloud(temp_file, pcd)
            
            try:
                # Convert superquadrics to format expected by grasp planning
                superquadric_list = []
                for sq in all_sqs:
                    euler_angles = R_simple.from_matrix(sq.RotM).as_euler('xyz')
                    superquadric_list.append({
                        'shape': sq.shape,
                        'scale': sq.scale,
                        'euler': euler_angles,
                        'translation': sq.translation
                    })
                
                # Generate grasps with debug visualization flags
                all_filtered_grasps = self.grasp_planning_manager.plan_grasps_multi_superquadric(
                    temp_file, superquadric_list
                )
            
                if all_filtered_grasps and len(all_filtered_grasps) > 0:
                    # Use the existing best grasp selection with criteria
                    object_center = np.mean(processed_points, axis=0)
                    best_grasp = self.grasp_planning_manager.planner.select_best_grasp_with_criteria(
                        all_filtered_grasps, object_center, processed_points
                    )
                    
                    # Extract all pose matrices
                    all_grasp_poses = []
                    for grasp in all_filtered_grasps:
                        if isinstance(grasp, dict) and 'pose' in grasp:
                            pose = grasp['pose']
                            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                                all_grasp_poses.append(pose)
                    
                    # Return all poses + best pose separately
                    best_pose = best_grasp['pose'] if best_grasp else None
                    return all_grasp_poses, best_pose
                else:
                    return [], None
                
            finally:
                # Cleanup temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            self.logger.error(f"Error generating grasp poses from superquadrics: {e}")
            return [], None
    
    def _visualize_superquadric_grasps(self, processed_points, all_sqs, all_grasp_poses, 
                                     best_grasp_pose, class_id):
        """Handle all superquadric grasp visualizations"""
        if not self.visualizer:
            return
            
        try:
            # Convert superquadric objects to visualization format
            superquadric_params = []
            for sq in all_sqs:
                euler_angles = R_simple.from_matrix(sq.RotM).as_euler('xyz')
                superquadric_params.append({
                    'shape': sq.shape,
                    'scale': sq.scale,
                    'euler': euler_angles,
                    'translation': sq.translation
                })
            
            # All valid grasps visualization
            if self.enable_all_valid_grasps_visualization:
                try:
                    self.visualizer.visualize_superquadric_grasps(
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
            if self.enable_best_grasps_visualization and best_grasp_pose is not None:
                try:
                    self.visualizer.visualize_superquadric_grasps(
                        point_cloud_data=processed_points,
                        superquadric_params=superquadric_params,
                        grasp_poses=[best_grasp_pose],  # Only the best grasp
                        show_sweep_volume=False,
                        window_name=f"Best Grasp: {self.class_names.get(class_id, f'Object_{class_id}')}"
                    )
                    self.logger.info(f"Best grasp visualization completed")
                except Exception as viz_error:
                    self.logger.error(f"Best grasp visualization failed: {viz_error}")
                    traceback.print_exc()
                    
        except Exception as e:
            self.logger.error(f"Error in superquadric grasp visualization: {e}")
    
    def _publish_best_grasp_pose(self, best_grasp_pose: np.ndarray, 
                            class_id: int, object_index: int):
        """Publish the best grasp pose"""
        try:
            if best_grasp_pose is None:
                self.logger.warning("No best grasp pose to publish")
                return
            
            # Ensure we have a 4x4 transformation matrix
            if not isinstance(best_grasp_pose, np.ndarray) or best_grasp_pose.shape != (4, 4):
                self.logger.error(f"Invalid grasp pose shape: {best_grasp_pose.shape if hasattr(best_grasp_pose, 'shape') else type(best_grasp_pose)}")
                return
            
            # Extract position and orientation
            pos = best_grasp_pose[:3, 3]
            rot_matrix = best_grasp_pose[:3, :3]
            quat = R_simple.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
            
            # Create pose message
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = self.node.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.target_frame
            
            pose_msg.pose.position.x = float(pos[0])
            pose_msg.pose.position.y = float(pos[1])
            pose_msg.pose.position.z = float(pos[2])
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])
            
            # Publish
            if self.pose_publisher:
                self.pose_publisher.publish(pose_msg)
                
                object_name = self.class_names.get(class_id, f'Class_{class_id}')
                self.logger.info(f"Published grasp pose for {object_name} at: "
                            f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
        except Exception as e:
            self.logger.error(f"Error publishing grasp pose: {e}")
            traceback.print_exc()

    # ============= COMMON METHODS =============
    
    def _grasp_execution_callback(self, msg):
        """Track grasp execution state"""
        self.grasp_executing = msg.data
        if msg.data:
            self.logger.info("Grasp execution started - pausing pose generation")
        else:
            self.logger.info("Grasp execution finished - resuming pose generation")
    
    def is_ready(self) -> bool:
        """Check if pose estimation manager is ready"""
        base_ready = self.is_initialized
        
        if self.method == 'icp':
            return base_ready and self.icp_estimator is not None
        elif self.method == 'superquadric':
            return base_ready and (not self.superquadric_enabled or self.grasp_planning_manager is not None)
        
        return base_ready
    
    def cleanup(self):
        """Clean up pose estimation resources"""
        try:
            if hasattr(self, 'grasp_planning_manager') and self.grasp_planning_manager:
                del self.grasp_planning_manager
            if hasattr(self, 'icp_estimator') and self.icp_estimator:
                del self.icp_estimator
            self.logger.info("Pose estimation manager cleaned up")
        except Exception as e:
            self.logger.error(f"Error during pose estimation cleanup: {e}")