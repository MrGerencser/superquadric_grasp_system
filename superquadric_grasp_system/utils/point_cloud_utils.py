import torch
import numpy as np
from typing import Tuple, List
import open3d as o3d


def crop_point_cloud_gpu(point_cloud: torch.Tensor, 
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


def convert_mask_to_3d_points(mask_indices: torch.Tensor, 
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


def fuse_point_clouds_centroid(point_clouds_cam1: List[Tuple[np.ndarray, int]],
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
            continue  # Skip this class_id entirely

        # If there is only one point cloud with the same class ID from each camera we can directly fuse the pcs
        if len(pcs1) == 1 and len(pcs2) == 1:
            # Concatenate the point clouds along the vertical axis
            fused_pc = filter_outliers_sor(np.vstack((pcs1[0], pcs2[0])))
            fused_point_clouds.append((fused_pc, class_id))

        # If there are multiple point clouds with the same class ID from each camera, we need to find the best match
        else:
            for pc1 in pcs1:
                pc1 = filter_outliers_sor(pc1)
                best_distance = float('inf')
                best_match = None

                # Calculate the centroid of the point cloud from camera 1
                centroid1 = calculate_centroid(pc1)

                # Loop over all the point clouds from camera 2 with the same ID and find the best match based on centroid distance
                for pc2 in pcs2:
                    centroid2 = calculate_centroid(pc2)
                    # Calculate the Euclidean distance / L2 norm between the centroids
                    distance = np.linalg.norm(centroid1 - centroid2)

                    if distance < best_distance and distance < distance_threshold:
                        best_distance = distance
                        best_match = pc2
                        best_match = filter_outliers_sor(best_match)

                # If a match was found, fuse the point clouds
                if best_match is not None:
                    # Concatenate the point clouds along the vertical axis and filter out the outliers
                    fused_pc = np.vstack((pc1, best_match))
                    fused_point_clouds.append((fused_pc, class_id))
                    # Remove the matched point cloud from the list of point clouds from camera 2 to prevent duplicate fusion
                    pcs2 = [pc for pc in pcs2 if not point_clouds_equal(pc, best_match)]

                # Only add unmatched point clouds if require_both_cameras is False
                elif not require_both_cameras:
                    fused_point_clouds.append((pc1, class_id))

            # Only add remaining camera 2 point clouds if require_both_cameras is False
            if not require_both_cameras:
                for pc2 in pcs2:
                    fused_point_clouds.append((pc2, class_id))

    return pcs1, pcs2, fused_point_clouds


def subtract_point_clouds_gpu(workspace_points: np.ndarray, object_points: np.ndarray, 
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
        print(f"Error in subtract_point_clouds_gpu: {e}")

def filter_outliers_sor(point_cloud: np.ndarray, 
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


def calculate_centroid(point_cloud: np.ndarray) -> np.ndarray:
    """Calculate the centroid (mean position) of a point cloud"""
    return np.mean(point_cloud, axis=0)


def point_clouds_equal(pc1: np.ndarray, 
                      pc2: np.ndarray) -> bool:
    """Check if two point clouds are equal"""
    return np.array_equal(pc1, pc2)


def downsample_point_cloud_gpu(point_cloud: torch.Tensor, 
                              voxel_size: float) -> torch.Tensor:
    """Downsample point cloud using GPU-accelerated voxel grid"""
    # Round the point cloud to the nearest voxel grid -> all points in the same voxel will have the same coordinates
    rounded = torch.round(point_cloud / voxel_size) * voxel_size
    downsampled_points = torch.unique(rounded, dim=0)
    return downsampled_points