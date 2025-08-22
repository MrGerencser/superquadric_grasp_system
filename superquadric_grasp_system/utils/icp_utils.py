import os
import numpy as np
import open3d as o3d
from typing import Optional, Dict, Any, Tuple, Union
from scipy.spatial.transform import Rotation as R_simple
from geometry_msgs.msg import PoseStamped, TransformStamped
import yaml


class ICPPoseEstimator:
    """ICP-based pose estimation utility class"""
    
    def __init__(self, model_folder_path: str, distance_threshold: float = 0.03, 
                visualize: bool = False, logger=None, class_names: Optional[Dict[int, str]] = None):
        self.model_folder_path = model_folder_path
        self.distance_threshold = distance_threshold
        self.visualize = visualize
        self.logger = logger
        
        # Reference models storage
        self.reference_models = {}
        self.processed_ref_points = {}
        
        # Class name mapping - use provided mapping or fallback to defaults
        self.class_names = class_names if class_names is not None else {
            0: "cone",
            1: "cup", 
            2: "power drill",
        }
        
        # Grasp poses storage
        self.grasp_poses = {}
        
        # Initialize once - remove duplicate loading
        self._initialize_models_and_grasps()
    
    def _initialize_models_and_grasps(self):
        """Initialize reference models and grasp poses"""
        if self.logger:
            self.logger.info("Initializing ICP pose estimator...")
            self.logger.debug(f"Class mapping: {self.class_names}")
            self.logger.debug(f"Model folder: {self.model_folder_path}")
        
        # Load models and grasp poses
        models_loaded, grasps_loaded = 0, 0
        
        for class_id, class_name in self.class_names.items():
            # Load model
            if self._load_single_model(class_id, class_name):
                models_loaded += 1
            
            # Load grasp poses
            if self._load_single_grasp_file(class_id, class_name):
                grasps_loaded += 1
        
        # Summary logging
        if self.logger:
            self.logger.info(f"ICP Estimator ready: {models_loaded}/{len(self.class_names)} models, {grasps_loaded}/{len(self.class_names)} grasp files")
            if models_loaded < len(self.class_names):
                missing_models = [f"{self.class_names[cid]}" for cid in self.class_names.keys() if cid not in self.reference_models]
                self.logger.warning(f"Missing models: {', '.join(missing_models)}")
    
    def _load_single_model(self, class_id: int, class_name: str) -> bool:
        """Load a single reference model"""
        try:
            # Try multiple paths
            model_paths = [
                os.path.join(self.model_folder_path, f"{class_name}.ply"),
                os.path.join(self.model_folder_path, class_name.replace(' ', '_'), f"{class_name.replace(' ', '_')}.ply"),
                os.path.join(self.model_folder_path, class_name.replace(' ', '_').lower(), f"{class_name.replace(' ', '_').lower()}.ply"),
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                if self.logger:
                    self.logger.warning(f"Model not found for {class_name} (class {class_id})")
                    self.logger.debug(f"Searched: {model_paths}")
                return False
            
            # Load and process model
            reference_model = o3d.io.read_point_cloud(model_path)
            if not reference_model.has_points():
                if self.logger:
                    self.logger.error(f"Empty model file: {model_path}")
                return False
            
            original_points = len(reference_model.points)
            
            # Scale conversion (mm to m)
            reference_model.scale(0.001, center=(0, 0, 0))
            
            # Downsample
            reference_model = reference_model.voxel_down_sample(voxel_size=0.003)
            final_points = len(reference_model.points)
            
            # Store processed model
            self.reference_models[class_id] = reference_model
            self.processed_ref_points[class_id] = np.asarray(reference_model.points).copy()
            
            if self.logger:
                self.logger.info(f"{class_name}: {original_points} -> {final_points} points (scaled mm->m)")
                self.logger.debug(f"File: {os.path.basename(model_path)}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load model for {class_name}: {e}")
            return False
    
    def _load_single_grasp_file(self, class_id: int, class_name: str) -> bool:
        """Load grasp poses for a single class"""
        try:
            # Try multiple grasp file paths
            grasp_file_paths = [
                os.path.join(self.model_folder_path, f"{class_name}_grasps.yaml"),
                os.path.join(self.model_folder_path, class_name.replace(' ', '_'), f"{class_name.replace(' ', '_')}_grasps.yaml"),
                os.path.join(self.model_folder_path, class_name.replace(' ', '_').lower(), f"{class_name.replace(' ', '_').lower()}_grasps.yaml"),
            ]
            
            grasp_file_path = None
            for path in grasp_file_paths:
                if os.path.exists(path):
                    grasp_file_path = path
                    break
            
            if not grasp_file_path:
                self.grasp_poses[class_id] = []
                if self.logger:
                    self.logger.debug(f"{class_name}: No grasp file found (will use default)")
                return False
            
            # Load grasp poses
            with open(grasp_file_path, 'r') as file:
                grasp_data = yaml.safe_load(file)
                self.grasp_poses[class_id] = grasp_data.get('grasps', [])
            
            grasp_count = len(self.grasp_poses[class_id])
            if self.logger:
                self.logger.info(f"{class_name}: {grasp_count} grasp poses loaded")
                self.logger.debug(f"File: {os.path.basename(grasp_file_path)}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load grasps for {class_name}: {e}")
            self.grasp_poses[class_id] = []
            return False
    
    def get_transformed_grasp_poses(self, transformation_matrix: np.ndarray, class_id: int, max_grasps: int = 3):
        """Transform pre-defined grasp poses using ICP result"""
        class_name = self.class_names.get(class_id, f'class_{class_id}')
        
        if self.logger:
            self.logger.debug(f"Transforming grasp poses for {class_name}")
        
        if class_id not in self.grasp_poses or not self.grasp_poses[class_id]:
            if self.logger:
                self.logger.info(f"No predefined grasps for {class_name}, generating default grasp")
            
            # Generate a simple default grasp at the object center
            default_grasp = {
                'name': 'default_top_grasp',
                'pose': transformation_matrix.copy(),
                'position': transformation_matrix[:3, 3].tolist(),
                'orientation': [0.0, 0.0, 0.0]  # No additional rotation
            }
            return [default_grasp]
        
        transformed_grasps = []
        available_grasps = len(self.grasp_poses[class_id])
        
        for i, grasp in enumerate(self.grasp_poses[class_id][:max_grasps]):
            try:
                # Extract grasp pose relative to object
                rel_position = np.array(grasp['position'])
                rel_orientation = np.array(grasp['orientation'])
                
                # Create relative transformation matrix
                rel_rotation = R_simple.from_euler('xyz', rel_orientation).as_matrix()
                rel_transform = np.eye(4)
                rel_transform[:3, :3] = rel_rotation
                rel_transform[:3, 3] = rel_position
                
                # Transform to world coordinates
                world_transform = transformation_matrix @ rel_transform
                
                transformed_grasps.append({
                    'name': grasp['name'],
                    'pose': world_transform,
                    'position': world_transform[:3, 3].tolist(),
                    'orientation': R_simple.from_matrix(world_transform[:3, :3]).as_euler('xyz').tolist()
                })
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to transform grasp {grasp.get('name', f'#{i+1}')}: {e}")
                continue
        
        if self.logger:
            self.logger.info(f"Generated {len(transformed_grasps)}/{available_grasps} grasp poses for {class_name}")
        
        return transformed_grasps
    
    def estimate_pose(self, observed_data: Union[o3d.geometry.PointCloud, np.ndarray], 
                     class_id: int, full_cloud: Optional[Union[o3d.geometry.PointCloud, np.ndarray]] = None) -> Optional[np.ndarray]:
        """Estimate pose using ICP alignment"""
        class_name = self.class_names.get(class_id, f'class_{class_id}')
        
        if self.logger:
            self.logger.debug(f"Starting ICP pose estimation for {class_name}")
        
        try:
            # Convert input to Open3D point cloud if it's a numpy array
            if isinstance(observed_data, np.ndarray):
                if len(observed_data.shape) != 2 or observed_data.shape[1] != 3:
                    if self.logger:
                        self.logger.error(f"Invalid point cloud shape: {observed_data.shape}, expected (N, 3)")
                    return None
                
                observed_cloud = o3d.geometry.PointCloud()
                observed_cloud.points = o3d.utility.Vector3dVector(observed_data)
            else:
                observed_cloud = observed_data
            
            if len(observed_cloud.points) < 10:
                if self.logger:
                    self.logger.warning("Insufficient points for ICP")
                return None
            
            # Get reference model for this class
            if class_id not in self.reference_models:
                if self.logger:
                    self.logger.error(f"No reference model for {class_name}")
                return None
            
            reference_points = self.processed_ref_points[class_id].copy()
            reference_copy = o3d.geometry.PointCloud()
            reference_copy.points = o3d.utility.Vector3dVector(reference_points)
            
            # Target center
            target_center = observed_cloud.get_center()
            
            # Convert full_cloud to Open3D if it's provided as numpy array
            if full_cloud is not None:
                if isinstance(full_cloud, np.ndarray):
                    full_cloud_o3d = o3d.geometry.PointCloud()
                    full_cloud_o3d.points = o3d.utility.Vector3dVector(full_cloud)
                    full_cloud = full_cloud_o3d
                
                # Crop full cloud for ICP target
                ref_size = np.array(reference_copy.get_max_bound()) - np.array(reference_copy.get_min_bound())
                cropped_cloud = full_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=target_center - ref_size * 1.5,
                    max_bound=target_center + ref_size * 1.5
                ))
            else:
                cropped_cloud = observed_cloud
            
            # Compute PCA bases
            obs_evals, obs_basis = self._get_pca_basis(np.asarray(observed_cloud.points))
            ref_evals, ref_basis = self._get_pca_basis(reference_points)
            
            if self.logger:
                self.logger.info(f"Class {class_id} - Observed eigenvalues: {obs_evals}")
                self.logger.info(f"Class {class_id} - Reference eigenvalues: {ref_evals}")
            
            # Try different PCA alignments
            configs = [
                {"flip1": 1, "flip2": 1},
                {"flip1": 1, "flip2": -1},
                {"flip1": -1, "flip2": 1},
                {"flip1": -1, "flip2": -1}
            ]
            
            best_result = {
                "fitness": 0.0,
                "rmse": float("inf"),
                "transformation": None,
                "config": "",
                "referencgit fetch origine": None,
                "threshold": self.distance_threshold
            }
            
            for i, cfg in enumerate(configs):
                try:
                    # Create initial alignment using PCA
                    R_pca = np.column_stack([
                        obs_basis[:, 0],
                        cfg["flip1"] * obs_basis[:, 1],
                        cfg["flip2"] * obs_basis[:, 2]
                    ]) @ ref_basis.T
                    
                    # FIX: Ensure right-handed coordinate system
                    det = np.linalg.det(R_pca)
                    if det < 0:
                        if self.logger:
                            self.logger.warning(f"Config {i}: Left-handed coordinate system detected (det={det:.6f}), fixing...")
                        # Flip the third column to make it right-handed
                        R_pca[:, 2] *= -1
                        det = np.linalg.det(R_pca)
                        if self.logger:
                            self.logger.info(f"Config {i}: Fixed determinant: {det:.6f}")
                    
                    # Ensure it's a proper rotation matrix (orthogonal)
                    U, _, Vt = np.linalg.svd(R_pca)
                    R_pca = U @ Vt
                    
                    # Double check the determinant after SVD correction
                    det_final = np.linalg.det(R_pca)
                    if det_final < 0:
                        if self.logger:
                            self.logger.warning(f"Config {i}: Still left-handed after SVD (det={det_final:.6f}), flipping...")
                        R_pca[:, 2] *= -1
                        det_final = np.linalg.det(R_pca)
                    
                    if self.logger:
                        self.logger.debug(f"Config {i}: Final determinant: {det_final:.6f}")
                    
                    # Initial transformation
                    T_init = np.eye(4)
                    T_init[:3, :3] = R_pca
                    T_init[:3, 3] = target_center - R_pca @ np.mean(reference_points, axis=0)
                    
                    # Apply initial transformation
                    ref_aligned = o3d.geometry.PointCloud(reference_copy)
                    ref_aligned.transform(T_init)
                    
                    # Point-to-point ICP
                    icp_result = o3d.pipelines.registration.registration_icp(
                        ref_aligned, cropped_cloud,
                        max_correspondence_distance=best_result["threshold"],
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                    )
                    
                    final_transform = icp_result.transformation @ T_init
                    
                    if icp_result.fitness > best_result["fitness"]:
                        best_result.update({
                            "fitness": icp_result.fitness,
                            "rmse": icp_result.inlier_rmse,
                            "transformation": final_transform,
                            "config": f"Config {i}: flip1={cfg['flip1']}, flip2={cfg['flip2']}, det={det_final:.3f}",
                            "reference": ref_aligned
                        })
                        
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"ICP config {i} failed: {e}")
                    continue
            
            # Point-to-plane refinement if good fitness
            if best_result["fitness"] > 0.4 and best_result["transformation"] is not None:
                try:
                    ref_refined = o3d.geometry.PointCloud(reference_copy)
                    ref_refined.transform(best_result["transformation"])
                    
                    # Estimate normals for point-to-plane ICP
                    cropped_cloud.estimate_normals()
                    
                    icp_p2plane = o3d.pipelines.registration.registration_icp(
                        ref_refined, cropped_cloud,
                        max_correspondence_distance=best_result["threshold"],
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                    )
                    
                    if icp_p2plane.fitness > best_result["fitness"]:
                        best_result["transformation"] = icp_p2plane.transformation @ best_result["transformation"]
                        best_result["fitness"] = icp_p2plane.fitness
                        best_result["rmse"] = icp_p2plane.inlier_rmse
                        
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Point-to-plane refinement failed: {e}")
            
            if best_result["transformation"] is None:
                if self.logger:
                    self.logger.warning(f"ICP failed for class {class_id}")
                return None
            
            # Success logging with status indicators
            if self.logger:
                fitness_status = "GOOD" if best_result['fitness'] > 0.7 else "OK" if best_result['fitness'] > 0.4 else "POOR"
                self.logger.info(f"{class_name}: {fitness_status} - Fitness={best_result['fitness']:.3f}, RMSE={best_result['rmse']:.4f}")
                self.logger.debug(f"Config: {best_result['config']}")
            
            # Visualization if enabled
            if self.visualize:
                self._visualize_icp_result(observed_cloud, reference_copy, best_result["transformation"], class_id)
            
            return best_result["transformation"]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ICP estimation failed for {class_name}: {e}")
            return None
    
    def _get_pca_basis(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PCA basis vectors sorted by descending eigenvalues"""
        cov = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]
    
    def _visualize_icp_result(self, observed_cloud: o3d.geometry.PointCloud, 
                             reference_cloud: o3d.geometry.PointCloud, 
                             transformation: np.ndarray, class_id: int):
        """Visualize ICP alignment result"""
        if not self.visualize:
            return
            
        try:
            # Create visualization
            aligned_ref = o3d.geometry.PointCloud(reference_cloud)
            aligned_ref.transform(transformation)
            
            # Color point clouds
            observed_cloud.paint_uniform_color([1, 0, 0])  # Red
            aligned_ref.paint_uniform_color([0, 0, 1])     # Blue
            
            # Display
            o3d.visualization.draw_geometries(
                [observed_cloud, aligned_ref],
                window_name=f"ICP Result - Class {class_id}",
                width=800,
                height=600
            )
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Visualization failed: {e}")
    
    def matrix_to_pose_stamped(self, transformation_matrix: np.ndarray, 
                              class_id: int, object_id: int, frame_id: str = "panda_link0") -> PoseStamped:
        """Convert transformation matrix to PoseStamped message"""
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = frame_id
        
        # Extract position
        pose_msg.pose.position.x = transformation_matrix[0, 3]
        pose_msg.pose.position.y = transformation_matrix[1, 3]
        pose_msg.pose.position.z = transformation_matrix[2, 3]
        
        # Extract rotation and convert to quaternion
        rotation_matrix = transformation_matrix[:3, :3]
        r = R_simple.from_matrix(rotation_matrix)
        quat = r.as_quat()  # [x, y, z, w]
        
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        return pose_msg
    
    def create_transform_stamped(self, transformation_matrix: np.ndarray, 
                                class_id: int, object_id: int, 
                                parent_frame: str = "panda_link0") -> TransformStamped:
        """Create TransformStamped message from transformation matrix"""
        transform = TransformStamped()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = f"object_{class_id}_{object_id}"
        
        # Position
        transform.transform.translation.x = transformation_matrix[0, 3]
        transform.transform.translation.y = transformation_matrix[1, 3]
        transform.transform.translation.z = transformation_matrix[2, 3]
        
        # Rotation
        rotation_matrix = transformation_matrix[:3, :3]
        r = R_simple.from_matrix(rotation_matrix)
        quat = r.as_quat()
        
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        return transform