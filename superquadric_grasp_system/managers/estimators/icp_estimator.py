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
                 max_grasp_candidates: int = 3, icp_grasp_planning_enabled: bool = True,
                 align_to_world_frame: bool = False):

        self.node = node
        self.logger = node.get_logger()
        
        # Configuration
        self.visualize_grasps = visualize_grasps
        self.max_grasp_candidates = max_grasp_candidates
        self.class_names = class_names
        self.icp_grasp_planning_enabled = icp_grasp_planning_enabled
        self.align_to_world_frame = align_to_world_frame

        # Debug log
        if self.logger:
            self.logger.info(f"ICP Grasp Planning Enabled: {self.icp_grasp_planning_enabled}")
            self.logger.info(f"Align to World Frame: {self.align_to_world_frame}")

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
        
        # Initialize visualizer if needed
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
        
        # Initialize ICP estimator
        self.icp_estimator = ICPPoseEstimator(
            model_folder_path=model_folder_path,
            distance_threshold=distance_threshold,
            visualize=visualize_alignment,
            align_to_world_frame=align_to_world_frame,
            icp_grasp_planning_enabled=icp_grasp_planning_enabled,
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
        """Process a single object point cloud using ICP"""
        # Preprocess point cloud
        processed_points = self.point_cloud_processor.preprocess(point_cloud, class_id)
        if processed_points is None:
            return None
        
        # Estimate pose
        pose_matrix = self.icp_estimator.estimate_pose(processed_points, class_id, workspace_cloud)
        if pose_matrix is None:
            return None
        
        # Store for visualization
        self._last_pose_matrix = pose_matrix
        if hasattr(self.icp_estimator, 'reference_models') and class_id in self.icp_estimator.reference_models:
            reference_model = self.icp_estimator.reference_models[class_id]
            self._transformed_reference_model = o3d.geometry.PointCloud(reference_model)
            self._transformed_reference_model.transform(pose_matrix)
        else:
            self._transformed_reference_model = None
        
        # Handle grasp planning
        if not self.icp_grasp_planning_enabled:
            # Publish raw pose only
            try:
                self.ros_publisher.publish_pose(pose_matrix, class_id, object_id)
                if self.logger:
                    self.logger.info("Grasp planning disabled for ICP - publishing raw pose only")
            except Exception as e:
                self.logger.error(f"Failed to publish ICP pose: {e}")
            
            return {
                'method': 'icp',
                'class_id': class_id,
                'pose_matrix': pose_matrix,
                'processed_points': processed_points,
                'grasp_poses': [],
                'best_grasp_pose': None,
                'object_id': object_id
            }
        
        # Generate and select grasp poses
        grasp_poses = self.icp_estimator.get_transformed_grasp_poses(
            pose_matrix, class_id, max_grasps=self.max_grasp_candidates
        )
        
        best_grasp_pose = self._select_best_grasp(grasp_poses, processed_points)
        
        # Visualize if enabled
        if self.visualize_grasps and self.visualizer:
            self._visualize_icp_grasps(processed_points, grasp_poses, best_grasp_pose, class_id)
        
        # Publish best grasp
        if best_grasp_pose is not None:
            try:
                self.ros_publisher.publish_pose(best_grasp_pose['pose'], class_id, object_id)
            except Exception as e:
                self.logger.error(f"Failed to publish ICP best-grasp pose: {e}")
        
        return {
            'method': 'icp',
            'class_id': class_id,
            'pose_matrix': pose_matrix,
            'processed_points': processed_points,
            'grasp_poses': grasp_poses,
            'best_grasp_pose': best_grasp_pose,
            'object_id': object_id
        }
    
    def _select_best_grasp(self, grasp_poses, point_cloud):
        """Select the best grasp pose using comprehensive scoring"""
        if not grasp_poses:
            return None
        
        robot_origin = np.array([0.0, 0.0, 0.0])
        preferred_approach = np.array([0.0, 0.0, 1.0])
        
        best_score = -float('inf')
        best_grasp = None
        best_info = None
        
        for grasp_idx, grasp in enumerate(grasp_poses):
            try:
                # Extract pose information
                pose = np.array(grasp['pose']) if 'pose' in grasp else np.eye(4)
                position = pose[:3, 3]
                rotation_matrix = pose[:3, :3]
                
                pos_x, pos_y, pos_z = float(position[0]), float(position[1]), float(position[2])
                position_scalar = np.array([pos_x, pos_y, pos_z])
                
                # Calculate approach direction
                gripper_z_world = rotation_matrix[:, 2]
                actual_approach_direction = -gripper_z_world
                approach_z = float(actual_approach_direction[2])
                
                # === SCORING COMPONENTS ===
                
                # 1. Distance scoring
                distance_to_base = np.linalg.norm(position_scalar - robot_origin)
                distance_score = 1.0 / (1.0 + distance_to_base)
                reach_penalty = max(0.0, (distance_to_base - 0.855) * 3.0)  # Max reach
                
                # 2. Height scoring
                if pos_z < 0.05:
                    height_score = 0.3
                elif pos_z > 0.2:
                    height_score = 0.7
                else:
                    height_score = 1.0
                
                # 3. Approach direction
                approach_alignment = np.dot(actual_approach_direction, preferred_approach)
                approach_score = max(0.0, approach_alignment)
                
                # 4. Below table penalty
                below_table_penalty = 0.0
                if approach_z < -0.5:
                    below_table_penalty = 5.0 * abs(approach_z)
                elif approach_z < 0.0:
                    below_table_penalty = 2.0 * abs(approach_z)
                
                # 5. Point cloud proximity
                point_cloud_distance_score = 0.0
                min_distance_to_cloud = None
                if point_cloud is not None:
                    try:
                        cloud_points = np.asarray(point_cloud.points) if hasattr(point_cloud, 'points') else point_cloud
                        distances = np.linalg.norm(cloud_points - position_scalar, axis=1)
                        min_distance_to_cloud = np.min(distances)
                        
                        if 0.01 <= min_distance_to_cloud <= 0.03:
                            point_cloud_distance_score = 1.0
                        elif min_distance_to_cloud > 0.03:
                            point_cloud_distance_score = 0.5
                        else:
                            point_cloud_distance_score = 0.2
                    except Exception:
                        pass
                
                # 6. Workspace scoring
                workspace_score = 0.0
                if 0.2 <= pos_x <= 0.6 and -0.3 <= pos_y <= 0.3:
                    workspace_score = 1.0
                elif 0.1 <= pos_x <= 0.7 and -0.4 <= pos_y <= 0.4:
                    workspace_score = 0.5
                
                # 7. Orientation feasibility
                orientation_score = 1.0
                try:
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_matrix(rotation_matrix)
                    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
                    
                    if abs(pitch) > np.pi/2.5:
                        orientation_score = 0.2
                    elif abs(pitch) > np.pi/4:
                        orientation_score = 0.6
                    
                    if abs(roll) > np.pi/3:
                        orientation_score *= 0.5
                except Exception:
                    orientation_score = 0.8
                
                # 8. Grasp type preference
                grasp_type_score = 1.0
                grasp_name = grasp.get('name', '').lower()
                if 'side' in grasp_name:
                    grasp_type_score = 1.2
                elif 'top' in grasp_name:
                    grasp_type_score = 1.0
                elif 'bottom' in grasp_name:
                    grasp_type_score = 0.1
                
                # Final score calculation
                final_score = (
                    1.0 * grasp_type_score +
                    2.0 * distance_score +
                    1.5 * height_score +
                    2.0 * approach_score +
                    1.0 * workspace_score +
                    1.0 * point_cloud_distance_score +
                    1.0 * orientation_score -
                    reach_penalty -
                    below_table_penalty
                )
                
                # Debug logging for first few grasps
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
                        'distance_score': distance_score,
                        'height_score': height_score,
                        'approach_score': approach_score,
                        'workspace_score': workspace_score,
                        'point_cloud_distance_score': point_cloud_distance_score,
                        'orientation_score': orientation_score,
                        'min_distance_to_cloud': min_distance_to_cloud,
                        'final_score': final_score
                    }
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error scoring grasp {grasp_idx+1}: {e}")
                continue
        
        # Log best selection
        if best_grasp is not None and self.logger:
            self.logger.info(f"\n[ICP SELECTION] Best grasp selected (#{best_info['grasp_index']+1}) with score: {best_score:.6f}")
            self.logger.info(f"   Grasp name: {best_grasp.get('name', 'unnamed')}")
            self.logger.info(f"   Distance score: {best_info['distance_score']:.3f}")
            self.logger.info(f"   Height score: {best_info['height_score']:.3f}")
            self.logger.info(f"   Approach score: {best_info['approach_score']:.3f}")
            self.logger.info(f"   Orientation score: {best_info['orientation_score']:.3f}")
            
            if best_info.get('min_distance_to_cloud') is not None:
                self.logger.info(f"   Min distance to point cloud: {best_info['min_distance_to_cloud']*1000:.1f}mm")
            
            if 'pose' in best_grasp:
                pos = best_grasp['pose'][:3, 3]
                self.logger.info(f"   FINAL POSITION: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        return best_grasp
    
    def _visualize_icp_grasps(self, processed_points, grasp_poses, best_grasp_pose, class_id):
        """Visualize ICP-based grasp poses"""
        if not grasp_poses or not self.visualizer:
            return
            
        try:
            all_poses = [g['pose'] for g in grasp_poses]
            best_pose = best_grasp_pose['pose'] if best_grasp_pose else None
            
            point_cloud_array = np.asarray(processed_points.points) if hasattr(processed_points, 'points') else processed_points
            class_name = self.class_names.get(class_id, f'Object_{class_id}')
            
            # Visualize all grasps
            self.visualizer.visualize_grasps(
                point_cloud_data=point_cloud_array,
                superquadric_params=None,
                grasp_poses=all_poses,
                gripper_colors=[(0, 0, 1)] * len(all_poses),
                show_sweep_volume=False,
                reference_model=getattr(self, '_transformed_reference_model', None),
                window_name=f"ICP Grasps: {class_name}"
            )
            
            # Visualize best grasp
            if best_pose is not None:
                self.visualizer.visualize_grasps(
                    point_cloud_data=point_cloud_array,
                    superquadric_params=None,
                    grasp_poses=[best_pose],
                    gripper_colors=[(0, 1, 0)],
                    show_sweep_volume=True,
                    reference_model=getattr(self, '_transformed_reference_model', None),
                    window_name=f"Best ICP Grasp: {class_name}"
                )
                
        except Exception as e:
            self.logger.error(f"ICP grasp visualization failed: {e}")
    
    def is_ready(self) -> bool:
        return self.icp_estimator is not None