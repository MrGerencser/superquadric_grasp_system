import numpy as np
import open3d as o3d

# ====================
# Configuration Section
# ====================
INPUT_PCD_PATH = "superquadric_grasp_system/models/object_models/cone/cone.ply"
OUTPUT_PCD_PATH = "superquadric_grasp_system/models/object_models/cone/cone_transformed.ply"
AXIS_MAPPING = [1, 2, 0]               # Mapping principal axes to X, Y, Z
# Use either SCALE or MEASURED_Z, not both
SCALE = 0.001                          # Scale factor for the point cloud
MEASURED_Z = None                      # Desired length for the Z axis after scaling in meter
FRAME_SIZE = 0.03                      # Size of coordinate frame for visualization
# ====================

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """
    Load a point cloud from a file.
    """
    return o3d.io.read_point_cloud(path)


def compute_principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the principal axes and centroid of the point cloud.

    Returns:
        axes (3x3 array): Columns are eigenvectors sorted by descending eigenvalue.
        centroid (3,): Mean position of the points.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = np.argsort(eigenvals)[::-1]
    axes = eigenvecs[:, order]
    return axes, centroid


def build_rotation_matrix(axes: np.ndarray, mapping: list[int]) -> np.ndarray:
    """
    Build a rotation matrix based on principal axes and custom mapping.

    Args:
        axes: 3x3 matrix of principal axes vectors.
        mapping: List of three indices mapping principal axes to X, Y, Z.
    Returns:
        Rotation matrix (3x3).
    """
    rot = np.column_stack([
        axes[:, mapping[0]],
        axes[:, mapping[1]],
        axes[:, mapping[2]]
    ]).T
    return rot


def align_and_scale(
    pcd: o3d.geometry.PointCloud,
    rotation: np.ndarray,
    centroid: np.ndarray,
    scale: float,
    target_z: float
) -> o3d.geometry.PointCloud:
    """
    Rotate, center, scale, and re-center the point cloud.
    """
    # Rotate around centroid
    pcd.rotate(rotation, center=centroid)

    # Move to origin
    pcd.translate(-centroid)

    # Compute scaling factor based on Z extent
    bounds = pcd.get_max_bound() - pcd.get_min_bound()
    current_z = bounds[2]
    if scale is not None:
        pcd.scale(scale, center=np.zeros(3))
    if target_z is not None:
        scale_factor = target_z / current_z
        pcd.scale(scale_factor, center=np.zeros(3))

    # Recentre at origin
    pcd.translate(-pcd.get_center())
    return pcd


def visualize(pcd: o3d.geometry.PointCloud, size: float):
    """
    Visualize the point cloud with a coordinate frame.
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    o3d.visualization.draw_geometries([pcd, frame])


def main():
    # Load point cloud
    pcd = load_point_cloud(INPUT_PCD_PATH)
    points = np.asarray(pcd.points)

    # Compute principal axes
    axes, centroid = compute_principal_axes(points)

    # Build rotation matrix
    rot_matrix = build_rotation_matrix(axes, AXIS_MAPPING)

    # Align and scale
    transformed_pcd = align_and_scale(pcd, rot_matrix, centroid, SCALE, MEASURED_Z)

    # Visualize
    visualize(transformed_pcd, FRAME_SIZE)

    # Save transformed point cloud
    o3d.io.write_point_cloud(
        OUTPUT_PCD_PATH,
        transformed_pcd,
        write_ascii=False,
        compressed=False
    )
    
    # Print dimensions and properties
    print("Dimensions of the transformed point cloud (m):")
    print("Width (X):", transformed_pcd.get_max_bound()[0] - transformed_pcd.get_min_bound()[0])
    print("Height (Y):", transformed_pcd.get_max_bound()[1] - transformed_pcd.get_min_bound()[1])
    print("Depth (Z):", transformed_pcd.get_max_bound()[2] - transformed_pcd.get_min_bound()[2])

    
if __name__ == "__main__":
    main()
