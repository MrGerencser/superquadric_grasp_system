#!/usr/bin/env python3
"""
Compute static tf2 from panda_link0 ➜ zed_left_camera_frame
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
# --- 1. 4×4 matrices (fill in from your YAML) -------------------------------
# T_chess_cam2z = np.array([[ 0.9958,  0.0341, -0.0853,  0.1346],
#                           [-0.0914,  0.4602, -0.8831,  0.7291],
#                           [ 0.0092,  0.8872,  0.4614, -0.3465],
#                           [ 0.0   ,  0.0   ,  0.0   ,  1.0   ]])

T_chess_cam2z = np.array([[ 0.9967,  0.0412, -0.0699,  0.0582],
                            [-0.0807,  0.5994, -0.7964,  0.5906],
                            [ 0.0091,  0.7994,  0.6007, -0.5129],
                            [ 0.0   ,  0.0   ,  0.0   ,  1.0   ]])


# T_chess_cam2z = np.array([[-0.9947, -0.0872,  0.0539,  0.0757],
#                             [ 0.1018, -0.7737,  0.6253, -0.6511],
#                             [-0.0128,  0.6275,  0.7785, -0.5198],
#                             [ 0.0   ,  0.0   ,  0.0   ,  1.0   ]])

# T_chess_cam2z = np.array([[ 0.9971,  0.0339, -0.0677,  0.0599],
#                           [-0.0745,  0.5995, -0.7969,  0.5903],
#                           [ 0.0136,  0.7996,  0.6003, -0.513 ],
#                           [ 0.0   ,  0.0   ,  0.0   ,  1.0   ]])


T_robot_chess = np.array([[-1.0 ,  0.0 ,  0.0 ,  0.358],
                          [ 0.0 ,  1.0 ,  0.0 ,  0.03 ],
                          [ 0.0 ,  0.0 , -1.0 ,  0.006],
                          [ 0.0 ,  0.0 ,  0.0 ,  1.0  ]])


T_cam2z_cam2x = np.array([[ 0.0,  -1.0,  1.0, 0.0],
                          [0.0,  0.0,  -1.0, 0.0],
                          [ 1.0, 0.0,  0.0, 0.0],
                          [ 0.0,  0.0,  0.0, 1.0]])



# --- 2. chain the transforms -----------------------------------------------
T_robot_cam2x = T_robot_chess @ T_chess_cam2z @ T_cam2z_cam2x
t = T_robot_cam2x[:3, 3]                              # translation
q = R.from_matrix(T_robot_cam2x[:3, :3]).as_quat()    # [x y z w]

# --- 3. print ros2 command --------------------------------------------------
print(
    f"ros2 run tf2_ros static_transform_publisher "
    f"--x {t[0]:.4f} --y {t[1]:.4f} --z {t[2]:.4f} "
    f"--qx {q[0]:.4f} --qy {q[1]:.4f} --qz {q[2]:.4f} --qw {q[3]:.4f} "
    f"--frame-id panda_link0 --child-frame-id zed_left_camera_frame"
)

