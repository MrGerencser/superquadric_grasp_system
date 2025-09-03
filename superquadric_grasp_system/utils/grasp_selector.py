import numpy as np
from typing import List, Dict, Optional, Union
import open3d as o3d


def quat_to_rot(quat):
    """Convert quaternion [x, y, z, w] to rotation matrix"""
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(quat).as_matrix()

def select_best_grasp(
    grasp_poses: List[Dict],
    point_cloud: Optional[Union[np.ndarray, o3d.geometry.PointCloud]] = None,
    workspace_bounds: Optional[Dict] = None,
    logger=None,
) -> Optional[Dict]:
    """
    Unified, minimal scorer (distance/height/workspace/downward) + optional point-cloud proximity.
    """
    if not grasp_poses:
        if logger:
            logger.warning("No grasp poses provided")
        return None

    if workspace_bounds is None:
        workspace_bounds = {
            'x_min': 0.2, 'x_max': 0.7,
            'y_min': -0.4, 'y_max': 0.4,
            'z_min': 0.01, 'z_max': 0.5
        }

    # Distance reference in XY-plane (from End_Effector)
    ee_base_xy = np.array([0.5, 0.0])

    # Prepare point cloud (optional)
    cloud_pts = None
    if point_cloud is not None:
        try:
            cloud_pts = np.asarray(point_cloud.points) if hasattr(point_cloud, "points") else np.asarray(point_cloud)
            if cloud_pts.ndim != 2 or cloud_pts.shape[1] != 3:
                if logger:
                    logger.warning("Point cloud has invalid shape; ignoring pc proximity term.")
                cloud_pts = None
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse point cloud; ignoring. Error: {e}")
            cloud_pts = None

    best_grasp, best_score = None, -1.0

    for grasp in grasp_poses:
        try:
            # --- Extract pose consistently ---
            if 'pose' in grasp:
                pose = np.array(grasp['pose'])
                position = pose[:3, 3]
                R_grasp = pose[:3, :3]
            elif 'position' in grasp and 'orientation' in grasp:
                position = np.array(grasp['position'])
                R_grasp = quat_to_rot(grasp['orientation'])
            else:
                if logger:
                    logger.warning(f'Grasp "{grasp.get("name", "unknown")}": invalid format (needs pose or position+orientation). Skipping.')
                continue

            # --- Core components (match simple function) ---
            pos_x, pos_y, pos_z = float(position[0]), float(position[1]), float(position[2])

            # 1) distance (XY only, to ee_base_xy)
            distance_to_base = np.linalg.norm(np.array([pos_x, pos_y]) - ee_base_xy)
            distance_score = 1.0 / (1.0 + distance_to_base)

            # 2) height
            height_score = 1.0 if pos_z > 0.05 else 0.3

            # 3) workspace soft gating
            in_ws = (
                workspace_bounds['x_min'] <= pos_x <= workspace_bounds['x_max'] and
                workspace_bounds['y_min'] <= pos_y <= workspace_bounds['y_max'] and
                workspace_bounds['z_min'] <= pos_z <= workspace_bounds['z_max']
            )
            workspace_score = 1.0 if in_ws else 0.3

            # 4) approach "downward" score (use gripper -Z alignment with world -Z)
            z_axis = R_grasp[:, 2]
            downward_score = max(0.0, -z_axis[2])

            # 5) optional: point-cloud proximity score (sweet spot 1–3 cm)
            pc_score = 0.0
            if cloud_pts is not None:
                try:
                    dists = np.linalg.norm(cloud_pts - position, axis=1)
                    dmin = float(dists.min())
                    if 0.01 <= dmin <= 0.03:
                        pc_score = 1.0
                    elif dmin > 0.03:
                        pc_score = 0.5
                    else:
                        pc_score = 0.2
                except Exception as e:
                    if logger:
                        logger.debug(f"PC proximity computation failed; ignoring pc term. Error: {e}")

            # --- Final score: identical weights + optional pc term ---
            final_score = (
                1.5 * distance_score +
                1.0 * height_score +
                2.0 * workspace_score +
                1.0 * downward_score +
                1.0 * pc_score  # only adds if cloud provided; else 0.0
            )

            if final_score > best_score:
                best_score = final_score
                best_grasp = grasp

            if logger:
                logger.debug(
                    f'Grasp "{grasp.get("name", "unknown")}" '
                    f'scores: dist={distance_score:.2f}, height={height_score:.2f}, '
                    f'workspace={workspace_score:.2f}, downward={downward_score:.2f}, '
                    f'pc={pc_score:.2f}, final={final_score:.2f}'
                )

        except Exception as e:
            if logger:
                logger.warning(f'Error scoring grasp "{grasp.get("name", "unknown")}": {e}')
            continue

    if best_grasp and logger:
        logger.info(f'Selected best grasp: "{best_grasp.get("name", "unknown")}" with score: {best_score:.3f}')
    return best_grasp
