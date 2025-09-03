from ...utils.icp_utils import ICPPoseEstimator
from ...utils.point_cloud_utils import PointCloudProcessor
from ...utils.ros_publisher import ROSPublisher
from ...utils.grasp_selector import select_best_grasp
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
        
        best_grasp_pose = select_best_grasp(
                                grasp_poses=grasp_poses,
                                point_cloud=point_cloud,
                                workspace_bounds=None,  # Use defaults
                                logger=self.logger,
                            )
        
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