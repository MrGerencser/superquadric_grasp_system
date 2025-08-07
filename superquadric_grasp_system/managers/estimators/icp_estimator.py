from ...utils.icp_utils import ICPPoseEstimator
from ...utils.point_cloud_utils import PointCloudProcessor
from ...utils.ros_publisher import ROSPublisher
import numpy as np
import open3d as o3d

class ICPEstimator:
    """ICP-specific pose estimation implementation"""
    
    def __init__(self, node, model_folder_path: str, distance_threshold: float,
                 max_iterations: int, convergence_threshold: float,
                 class_names: dict, poisson_reconstruction: bool,
                 outlier_removal: bool, voxel_downsample_size: float,
                 visualize_alignment: bool = False, visualize_grasps: bool = False,
                 max_grasp_candidates: int = 3):
        self.node = node
        self.logger = node.get_logger()
        
        # ICP-specific configuration
        self.model_folder_path = model_folder_path
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.visualize_alignment = visualize_alignment
        self.visualize_grasps = visualize_grasps
        self.max_grasp_candidates = max_grasp_candidates
        self.class_names = class_names
        
        # Create point cloud processor
        self.point_cloud_processor = PointCloudProcessor(
            poisson_reconstruction=poisson_reconstruction,
            outlier_removal=outlier_removal,
            voxel_downsample_size=voxel_downsample_size
        )
        
        # Create ROS publisher
        self.ros_publisher = ROSPublisher(
            node,
            {
                'class_names': class_names,
                'publish_poses': True,
                'publish_transforms': True,
                'target_frame': 'panda_link0'
            }
        )
        
        # Initialize visualizer if grasp visualization is enabled
        if self.visualize_grasps:
            try:
                from ...visualization.main_visualizer import PerceptionVisualizer
                self.visualizer = PerceptionVisualizer()
            except ImportError:
                self.logger.warning("Could not import visualizer, disabling visualization")
                self.visualizer = None
                self.visualize_grasps = False
        else:
            self.visualizer = None
        
        # Initialize ICP estimator using existing utility
        self.icp_estimator = ICPPoseEstimator(
            model_folder_path=self.model_folder_path,
            distance_threshold=self.distance_threshold,
            visualize=self.visualize_alignment,
            logger=self.logger,
            class_names=self.class_names
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
            grasp_poses = self.icp_estimator.get_transformed_grasp_poses(pose_matrix, class_id, max_grasps=self.max_grasp_candidates)
            
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
        """Select the best grasp pose from candidates with comprehensive criteria"""
        if not grasp_poses:
            return None
        
        # Define constants for scoring
        robot_origin = np.array([0.0, 0.0, 0.0])
        preferred_approach = np.array([0.0, 0.0, 1.0])  # Approach from above
        
        best_score = -float('inf')
        best_grasp = None
        best_info = None
        
        for grasp_idx, grasp in enumerate(grasp_poses):
            try:
                # Extract pose information
                if 'pose' in grasp:
                    pose = np.array(grasp['pose'])
                    position = pose[:3, 3]
                    rotation_matrix = pose[:3, :3]
                else:
                    position = np.array(grasp['position'])
                    rotation_matrix = np.eye(3)  # Default rotation if not available
                
                # Convert to scalars for calculations
                pos_x, pos_y, pos_z = float(position[0]), float(position[1]), float(position[2])
                position_scalar = np.array([pos_x, pos_y, pos_z])
                
                # Calculate approach direction
                gripper_z_world = rotation_matrix[:, 2]
                actual_approach_direction = -gripper_z_world  # Gripper approaches opposite to Z-axis
                approach_z = float(actual_approach_direction[2])
                
                # Base score (use a default for ICP grasps)
                base_score = 1.0  # Could be enhanced with grasp quality metrics
                
                # === SCORING COMPONENTS ===
                
                # 1. Distance and reach scoring
                distance_to_base = np.linalg.norm(position_scalar - robot_origin)
                distance_score = 1.0 / (1.0 + distance_to_base)
                
                max_reach = 0.855  # Franka Panda max reach
                reach_penalty = max(0.0, (distance_to_base - max_reach) * 3.0)
                
                # 2. Height preference scoring
                height_score = 1.0
                if pos_z < 0.05:  # Too low (near table)
                    height_score = 0.3
                elif pos_z > 0.2:  # Too high
                    height_score = 0.7
                elif 0.05 <= pos_z <= 0.15:  # Optimal height range
                    height_score = 1.0
                
                # 3. Approach direction preference
                approach_alignment = np.dot(actual_approach_direction, preferred_approach)
                approach_score = max(0.0, approach_alignment)
                
                # 4. Strong penalty for approaches from below table
                below_table_penalty = 0.0
                if approach_z < -0.5:  # Approaching from significantly below
                    below_table_penalty = 5.0 * abs(approach_z)  # Heavy penalty
                elif approach_z < 0.0:  # Any downward approach
                    below_table_penalty = 2.0 * abs(approach_z)  # Moderate penalty
                
                # 5. Point cloud distance scoring (proximity to object surface)
                point_cloud_distance_score = 0.0
                point_cloud_penalty = 0.0
                min_distance_to_cloud = None
                
                if point_cloud is not None:
                    try:
                        # Convert point cloud to numpy array if needed
                        if hasattr(point_cloud, 'points'):
                            cloud_points = np.asarray(point_cloud.points)
                        else:
                            cloud_points = point_cloud
                        
                        distances = np.linalg.norm(cloud_points - position_scalar, axis=1)
                        min_distance_to_cloud = np.min(distances)
                        
                        # Preference for grasps near object surface
                        if 0.01 <= min_distance_to_cloud <= 0.03:  # Sweet spot (1-3cm)
                            point_cloud_distance_score = 1.0
                        elif min_distance_to_cloud > 0.03:  # Too far from object
                            point_cloud_distance_score = 0.5
                        else:  # Very close < 1cm (potential collision)
                            point_cloud_distance_score = 0.2
                            
                    except Exception as pc_error:
                        if self.logger:
                            self.logger.warning(f"Point cloud distance calculation failed: {pc_error}")
                
                # 6. Workspace preference scoring
                workspace_score = 0.0
                if 0.2 <= pos_x <= 0.6 and -0.3 <= pos_y <= 0.3:  # Optimal workspace
                    workspace_score = 1.0
                elif 0.1 <= pos_x <= 0.7 and -0.4 <= pos_y <= 0.4:  # Acceptable workspace
                    workspace_score = 0.5
                
                # 7. Orientation feasibility (additional check for ICP)
                orientation_score = 1.0
                try:
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_matrix(rotation_matrix)
                    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
                    
                    # Penalize extreme orientations
                    if abs(pitch) > np.pi/2.5:  # > 72 degrees
                        orientation_score = 0.2
                    elif abs(pitch) > np.pi/4:  # > 45 degrees
                        orientation_score = 0.6
                    
                    if abs(roll) > np.pi/3:  # > 60 degrees
                        orientation_score *= 0.5
                        
                except Exception:
                    orientation_score = 0.8  # Default if orientation check fails
                
                # 8. Grasp type preference (from name if available)
                grasp_type_score = 1.0
                grasp_name = grasp.get('name', '').lower()
                if 'side' in grasp_name:
                    grasp_type_score = 1.2  # Prefer side grasps
                elif 'top' in grasp_name:
                    grasp_type_score = 1.0   # Top grasps are neutral
                elif 'bottom' in grasp_name:
                    grasp_type_score = 0.1   # Avoid bottom grasps
                
                # === FINAL SCORE CALCULATION ===
                final_score = (
                    base_score * grasp_type_score +           # Base quality with type preference
                    2.0 * distance_score +                    # Prefer closer grasps
                    1.5 * height_score +                      # Height preference
                    2.0 * approach_score +                    # Approach direction preference
                    1.0 * workspace_score +                   # Workspace preference
                    1.0 * point_cloud_distance_score +        # Object proximity preference
                    1.0 * orientation_score +                 # Orientation feasibility
                    -reach_penalty -                          # Penalties
                    below_table_penalty
                )
                
                # Debug output for first few grasps
                if grasp_idx < 3 and self.logger:
                    self.logger.info(f"  [ICP SCORE] Grasp {grasp_idx+1} ({grasp.get('name', 'unnamed')}):")
                    self.logger.info(f"    Position: [{pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f}]")
                    self.logger.info(f"    Approach Z: {approach_z:.3f}")
                    self.logger.info(f"    Distance to base: {distance_to_base:.3f}m")
                    if min_distance_to_cloud is not None:
                        self.logger.info(f"    Min distance to cloud: {min_distance_to_cloud*1000:.1f}mm")
                    self.logger.info(f"    Final score: {final_score:.6f}")
                
                # Track best grasp
                if final_score > best_score:
                    best_score = final_score
                    best_grasp = grasp
                    best_info = {
                        'grasp_index': grasp_idx,
                        'base_score': base_score,
                        'distance_score': distance_score,
                        'height_score': height_score,
                        'approach_score': approach_score,
                        'workspace_score': workspace_score,
                        'point_cloud_distance_score': point_cloud_distance_score,
                        'orientation_score': orientation_score,
                        'grasp_type_score': grasp_type_score,
                        'min_distance_to_cloud': min_distance_to_cloud,
                        'final_score': final_score
                    }
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error scoring grasp {grasp_idx+1}: {e}")
                continue
        
        # Log best grasp selection
        if best_grasp is not None and self.logger:
            self.logger.info(f"\n[ICP SELECTION] Best grasp selected (#{best_info['grasp_index']+1}) with score: {best_score:.6f}")
            self.logger.info(f"   Grasp name: {best_grasp.get('name', 'unnamed')}")
            self.logger.info(f"   Distance score: {best_info['distance_score']:.3f}")
            self.logger.info(f"   Height score: {best_info['height_score']:.3f}")
            self.logger.info(f"   Approach score: {best_info['approach_score']:.3f}")
            self.logger.info(f"   Orientation score: {best_info['orientation_score']:.3f}")
            
            if best_info.get('min_distance_to_cloud') is not None:
                self.logger.info(f"   Min distance to point cloud: {best_info['min_distance_to_cloud']*1000:.1f}mm")
            
            # Show final pose
            if 'pose' in best_grasp:
                best_pose = best_grasp['pose']
                pos = best_pose[:3, 3]
                self.logger.info(f"   FINAL POSITION: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        else:
            if self.logger:
                self.logger.warning("No feasible ICP grasp found - all candidates scored poorly")
        
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
    
    def _needs_visualization(self) -> bool:
        return (
            self.visualize_grasps or  # ICP-specific grasp visualization
            self.visualize_alignment   # ICP alignment visualization
        )
    
    def is_ready(self) -> bool:
        return self.icp_estimator is not None