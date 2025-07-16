import torch
import numpy as np
from typing import Tuple, List
import open3d as o3d

import rclpy.logging

class PointCloudProcessor:
    """Handles shared point cloud preprocessing operations"""
    
    def __init__(self, config):
        self.poisson_reconstruction = config.get('poisson_reconstruction', False)
        self.outlier_removal = config.get('outlier_removal', True)
        self.voxel_downsample_size = config.get('voxel_downsample_size', 0.002)
        
        # Visualization settings
        self.enable_detected_object_clouds_visualization = config.get('enable_detected_object_clouds_visualization', False)
        self.visualizer = None  # Will be set by the estimator if needed
        
        self.logger = rclpy.logging.get_logger('point_cloud_processor')

    def set_visualization_components(self, visualizer, *args, **kwargs):
        """Set visualization components from the estimator"""
        self.visualizer = visualizer
    
    def preprocess(self, point_cloud, class_id=None):
        """Unified preprocessing for both ICP and superquadric methods"""
        try:
            # Convert numpy array to appropriate format if needed
            if isinstance(point_cloud, np.ndarray):
                if len(point_cloud.shape) != 2 or point_cloud.shape[1] != 3:
                    self.logger.error(f"Invalid point cloud shape: {point_cloud.shape}")
                    return None
            else:
                # Convert other formats to numpy array
                point_cloud = np.asarray(point_cloud)
            
            # Store original points for visualization
            original_object_points = point_cloud.copy()
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            
            # Apply outlier removal if enabled
            if self.outlier_removal:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
                pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
            
            # Downsample
            if self.voxel_downsample_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_downsample_size)
            
            if len(pcd.points) < 50:  # Minimum threshold for both methods
                if self.logger:
                    self.logger.warning(f"Too few points after preprocessing: {len(pcd.points)}")
                return None
            
            filtered_object_points = np.asarray(pcd.points)
            
            # Apply Poisson reconstruction if enabled (shared)
            if self.poisson_reconstruction:
                filtered_object_points = self.apply_poisson_reconstruction(
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
                    if self.logger:
                        self.logger.warning(f"Detected cloud filtering visualization failed: {viz_error}")
            
            return filtered_object_points
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error preprocessing point cloud: {e}")
            return None
    
    # ... rest of your existing methods
    
    def apply_poisson_reconstruction(self, pcd, filtered_points, original_points):
        """Apply Poisson reconstruction"""
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
        
        # Always fall back to filtered points
        return filtered_points
    

    def crop_point_cloud_gpu(self, point_cloud: torch.Tensor, 
                            x_bounds: Tuple[float, float],
                            y_bounds: Tuple[float, float], 
                            z_bounds: Tuple[float, float]) -> torch.Tensor:
        """Crop point cloud to workspace bounds using GPU"""
        mask = (
            (point_cloud[:, 0] >= x_bounds[0]) & (point_cloud[:, 0] <= x_bounds[1]) &
            (point_cloud[:, 1] >= y_bounds[0]) & (point_cloud[:, 1] <= y_bounds[1]) &
            (point_cloud[:, 2] >= z_bounds[0]) & (point_cloud[:, 2] <= z_bounds[1])
        )
        return point_cloud[mask]


    def convert_mask_to_3d_points(self, mask_indices: torch.Tensor, 
                                depth_map: torch.Tensor,
                                cx: float, 
                                cy: float, 
                                fx: float, 
                                fy: float) -> torch.Tensor:
        """Convert 2D mask to 3D points using camera intrinsics"""
        u_coords = mask_indices[:, 1]
        v_coords = mask_indices[:, 0]
        
        # Extract the necessary depth values from the depth map
        depth_values = depth_map[v_coords, u_coords]
        
        # Create a mask to filter out invalid depth values
        valid_mask = (depth_values > 0) & ~torch.isnan(depth_values) & ~torch.isinf(depth_values)
        
        # Filter out invalid depth values
        u_coords = u_coords[valid_mask]
        v_coords = v_coords[valid_mask]
        depth_values = depth_values[valid_mask]
        
        # Calculate the 3D coordinates using the camera intrinsics
        x_coords = (u_coords - cx) * depth_values / fx
        y_coords = (v_coords - cy) * depth_values / fy
        z_coords = depth_values
        
        # Return a tensor of shape (N, 3) containing the 3D coordinates
        return torch.stack((x_coords, y_coords, z_coords), dim=-1)


    def fuse_point_clouds_centroid(self, point_clouds_cam1: List[Tuple[np.ndarray, int]],
                                point_clouds_cam2: List[Tuple[np.ndarray, int]],
                                distance_threshold: float = 0.3,
                                require_both_cameras: bool = True) -> Tuple[List, List, List]:
        """Fuse point clouds from two cameras based on centroid distance"""
        # Group the point clouds by the class ID
        # The class dicts are of the form: {class_id1: [point_cloud1, point_cloud2, ...], class_id2: [point_cloud1, point_cloud2, ...]}
        pcs1 = []
        pcs2 = []
        class_dict1 = {}
        class_dict2 = {}

        # Iterate over the point clouds from camera 1 and group them by class ID
        # point_clouds_camera1 = [(pc, class_id), ...] is a list of tuples containing the point cloud and the class ID
        for pc, class_id in point_clouds_cam1:
            if class_id not in class_dict1:
                class_dict1[class_id] = []  # To store the point clouds for the class ID
            class_dict1[class_id].append(pc)  # Append the point cloud to the list for the class ID

        # Iterate over the point clouds from camera 2 and group them by class ID
        for pc, class_id in point_clouds_cam2:
            if class_id not in class_dict2:
                class_dict2[class_id] = []
            class_dict2[class_id].append(pc)

        # After this loop, class_dict1 and class_dict2 contain the point clouds grouped by class ID

        # Initialize the fused point cloud list
        fused_point_clouds = []

        # Process each class ID
        # Get all the unique class IDs from both cameras
        # class_dict1.keys() returns a set of all the keys "class IDs" in the dictionary
        for class_id in set(class_dict1.keys()).union(class_dict2.keys()):
            # Get the point clouds for the current class ID from both cameras
            pcs1 = class_dict1.get(class_id, [])  # pcs1 has the following format: [point_cloud1, point_cloud2, ...]
            pcs2 = class_dict2.get(class_id, [])

            # Skip objects that are only detected by one camera
            if require_both_cameras and (len(pcs1) == 0 or len(pcs2) == 0):
                self.logger.debug(f"Skipping class ID {class_id} as it is only detected by one camera.")    
                continue  # Skip this class_id entirely

            # If there is only one point cloud with the same class ID from each camera we can directly fuse the pcs
            if len(pcs1) == 1 and len(pcs2) == 1:
                # Concatenate the point clouds along the vertical axis
                fused_pc = self.filter_outliers_sor(np.vstack((pcs1[0], pcs2[0])))
                fused_point_clouds.append((fused_pc, class_id))

            # If there are multiple point clouds with the same class ID from each camera, we need to find the best match
            else:
                for pc1 in pcs1:
                    pc1 = self.filter_outliers_sor(pc1)
                    best_distance = float('inf')
                    best_match = None

                    # Calculate the centroid of the point cloud from camera 1
                    centroid1 = self.calculate_centroid(pc1)

                    # Loop over all the point clouds from camera 2 with the same ID and find the best match based on centroid distance
                    for pc2 in pcs2:
                        centroid2 = self.calculate_centroid(pc2)
                        # Calculate the Euclidean distance / L2 norm between the centroids
                        distance = np.linalg.norm(centroid1 - centroid2)

                        if distance < best_distance and distance < distance_threshold:
                            best_distance = distance
                            best_match = pc2
                            best_match = self.filter_outliers_sor(best_match)

                    # If a match was found, fuse the point clouds
                    if best_match is not None:
                        # Concatenate the point clouds along the vertical axis and filter out the outliers
                        fused_pc = np.vstack((pc1, best_match))
                        fused_point_clouds.append((fused_pc, class_id))
                        # Remove the matched point cloud from the list of point clouds from camera 2 to prevent duplicate fusion
                        pcs2 = [pc for pc in pcs2 if not self.point_clouds_equal(pc, best_match)]

                    # Only add unmatched point clouds if require_both_cameras is False
                    elif not require_both_cameras:
                        fused_point_clouds.append((pc1, class_id))

                # Only add remaining camera 2 point clouds if require_both_cameras is False
                if not require_both_cameras:
                    for pc2 in pcs2:
                        fused_point_clouds.append((pc2, class_id))

        return pcs1, pcs2, fused_point_clouds


    def subtract_point_clouds_gpu(self, workspace_points: np.ndarray, object_points: np.ndarray, 
                                distance_threshold: float = 0.06, batch_size: int = 10000) -> np.ndarray:
        """
        Subtract object points from workspace points using GPU acceleration with batching
        """
        if workspace_points.size == 0:
            return workspace_points
        
        if object_points.size == 0:
            return workspace_points
            
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Convert to torch tensors
            workspace_torch = torch.tensor(workspace_points, dtype=torch.float32, device=device)
            object_torch = torch.tensor(object_points, dtype=torch.float32, device=device)
            
            # Process in batches to avoid memory explosion
            keep_mask = torch.ones(workspace_torch.shape[0], dtype=torch.bool, device=device)
            
            # Process workspace points in batches
            for i in range(0, workspace_torch.shape[0], batch_size):
                end_idx = min(i + batch_size, workspace_torch.shape[0])
                workspace_batch = workspace_torch[i:end_idx]
                
                # For each workspace batch, check against all object points in sub-batches
                batch_keep = torch.ones(workspace_batch.shape[0], dtype=torch.bool, device=device)
                
                for j in range(0, object_torch.shape[0], batch_size):
                    obj_end_idx = min(j + batch_size, object_torch.shape[0])
                    obj_batch = object_torch[j:obj_end_idx]
                    
                    # Compute distances: [workspace_batch_size, obj_batch_size]
                    distances = torch.cdist(workspace_batch, obj_batch, p=2)
                    
                    # Check if any object point is within threshold
                    min_distances = torch.min(distances, dim=1)[0]
                    batch_keep = batch_keep & (min_distances > distance_threshold)
                    
                    # Clean up intermediate tensors
                    del distances, min_distances
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                
                keep_mask[i:end_idx] = batch_keep
                
                # Clean up batch tensors
                del workspace_batch, batch_keep
                if device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Filter and return
            filtered_points = workspace_torch[keep_mask]
            result = filtered_points.cpu().numpy()
            
            # Clean up
            del workspace_torch, object_torch, keep_mask, filtered_points
            if device == 'cuda':
                torch.cuda.empty_cache()
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in subtract_point_clouds_gpu: {e}")

    def filter_outliers_sor(self, point_cloud: np.ndarray, 
                        nb_neighbors: int = 20, 
                        std_ratio: float = 1.5) -> np.ndarray:
        """Apply statistical outlier removal to filter noise from point cloud"""
        # Convert the numpy array to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # Apply statistical outlier removal
        filtered_pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio
        )

        # Convert back to numpy array
        filtered_points = np.asarray(filtered_pcd.points)
        return filtered_points


    def calculate_centroid(self, point_cloud: np.ndarray) -> np.ndarray:
        """Calculate the centroid (mean position) of a point cloud"""
        return np.mean(point_cloud, axis=0)


    def point_clouds_equal(self, pc1: np.ndarray, 
                        pc2: np.ndarray) -> bool:
        """Check if two point clouds are equal"""
        return np.array_equal(pc1, pc2)


    def downsample_point_cloud_gpu(self, point_cloud: torch.Tensor, 
                                voxel_size: float) -> torch.Tensor:
        """Downsample point cloud using GPU-accelerated voxel grid"""
        # Round the point cloud to the nearest voxel grid -> all points in the same voxel will have the same coordinates
        rounded = torch.round(point_cloud / voxel_size) * voxel_size
        downsampled_points = torch.unique(rounded, dim=0)
        return downsampled_points