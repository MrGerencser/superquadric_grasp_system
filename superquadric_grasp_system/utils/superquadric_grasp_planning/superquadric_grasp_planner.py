import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R_simple

import rclpy.logging

from .geometric_primitives import Superquadric, Gripper, rotation_from_u_to_v
from .grasp_filters import support_test, collision_test, score_grasp, sample_superquadric_surface

DEG = np.pi / 180.0

# ------------------------------------------------------------
# SuperquadricGraspPlanner class 
# ------------------------------------------------------------

class SuperquadricGraspPlanner:
    def __init__(self, jaw_len=0.054, max_open=0.08):
        """
        Initialize the superquadric-based grasp planner
        
        Args:
            jaw_len: gripper jaw length in meters
            max_open: maximum gripper opening in meters
        """
        self.gripper = Gripper(jaw_len=jaw_len, max_open=max_open)
        # Store last valid grasps for visualization
        self.last_valid_grasps = []
        
        self.logger = rclpy.logging.get_logger('superquadric_grasp_planner')

    def principal_axis_sweeps(self, S: Superquadric, G: Gripper, step_deg=10):
        candidate_R = []

        self.logger.debug(f"This implementation assumes gripper closing line λ is along Y-axis (not X-axis as in original paper)")
        self.logger.debug(f"Principal axes in world frame:")
        self.logger.debug(f"  λ_x: {S.axes_world[:, 0]}")
        self.logger.debug(f"  λ_y: {S.axes_world[:, 1]}")
        self.logger.debug(f"  λ_z: {S.axes_world[:, 2]}")
        self.logger.debug(f"Gripper closing line λ: {G.lambda_local}")

        # For each principal axis λi (i = x,y,z)
        for i, axis_world in enumerate(S.axes_world.T):  # iterate over λ_x, λ_y, λ_z
            axis_names = ["λ_x", "λ_y", "λ_z"]
            axis_name = axis_names[i]

            self.logger.debug(f"Creating set Δ{axis_name}:")
            self.logger.debug(f"  Step 1: Align gripper closing line λ with {axis_name} = {axis_world}")

            # Step 1: Compute Rotation that aligns gripper's closing line λ → principal axis
            R_align = rotation_from_u_to_v(G.lambda_local, axis_world)

            self.logger.debug(f"  Step 2: Rotate gripper around {axis_name} every {step_deg}°")

            # Step 2: Rotate the gripper around the aligned axis in step_deg increments
            rotations_in_set = 0
            for theta_deg in range(0, 360, step_deg):
                theta_rad = theta_deg * DEG
                
                # Create rotation around the principal axis
                R_spin = R_simple.from_rotvec(axis_world * theta_rad).as_matrix()
                
                # Combined rotation: first align, then spin
                R_final = R_spin @ R_align
                
                candidate_R.append(R_final)
                rotations_in_set += 1
                
                # Debug first few rotations
                if theta_deg < 30:  # Show first 3 rotations for verification
                    closing_dir_after = R_final @ G.lambda_local
                    alignment_check = np.dot(closing_dir_after, axis_world)
                    self.logger.debug(f"    θ={theta_deg:3d}°: closing_dir={closing_dir_after}, alignment={alignment_check:.3f}")

            self.logger.debug(f"  → Created set Δ{axis_name} with {rotations_in_set} poses")

        expected_total = 3 * (360 // step_deg)  # 3 axes × (360/step_deg) rotations
        self.logger.debug(f"principal axis sweep complete: {len(candidate_R)} candidates (expected: {expected_total})")

        return candidate_R

    def extra_sweeps_special_shapes(self, S: Superquadric, base_R_list, G: Gripper):
        """
        For prism-like (ε1→0) or cuboid-like (ε2→0): slide gripper along SQ-local axes in 15mm grid.
        For cylinder-like (ε1→0, ε2=1, ax=ay): rotate about λ_z in π/8 increments.
        
        NOTE: This implementation assumes gripper closing line λ is along Y-axis (not X-axis as in original paper)
        """
        R_list_full = list(base_R_list)
        poses_offsets = []

        # Heuristic thresholds
        prism_like   = (S.ε1 < 0.3)  # ε1 → 0
        cuboid_like  = (S.ε2 < 0.3)  # ε2 → 0
        
        # cylinder condition (most specific, check first)
        cylinder_like = (S.ε1 < 0.3) and (abs(S.ε2 - 1.0) < 0.1) and (abs(S.ax - S.ay) < 0.01)

        self.logger.debug(f"Shape analysis: ε1={S.ε1:.3f}, ε2={S.ε2:.3f}, ax={S.ax:.3f}, ay={S.ay:.3f}")
        self.logger.debug(f"prism_like={prism_like}, cuboid_like={cuboid_like}, cylinder_like={cylinder_like}")
        self.logger.debug(f"Gripper closing line λ is along: {G.lambda_local} (Y-axis)")

        # SQ-local axes in world
        λx_w, λy_w, λz_w = S.axes_world[:, 0], S.axes_world[:, 1], S.axes_world[:, 2]

        # Grid step (15 mm)
        step = 0.015

        def linspace_clamped(extent):
            if extent < 1e-6:
                return np.array([0.0])
            n = int(np.floor(extent / step))
            vals = np.arange(-n, n+1) * step
            return vals

        # Handle cylinder case FIRST (most specific)
        if cylinder_like:
            self.logger.debug(f"[INFO] Detected cylinder shape - adding π/8 rotations around z-axis")

            # For cylinders: only rotate around z-axis (no translation)
            seen = set()
            axis_world = λz_w
            R_align = rotation_from_u_to_v(G.lambda_local, axis_world)
            
            # π/8 = 22.5 degrees intervals as per original paper
            for extra_angle in np.arange(0, 360, 22.5):
                R_extra = R_simple.from_rotvec(axis_world * (extra_angle * DEG)).as_matrix()
                R_new = R_extra @ R_align
                key = tuple(np.round(R_new, 6).ravel())
                if key not in seen:
                    seen.add(key)
                    R_list_full.append(R_new)
                    poses_offsets.append((R_new, np.zeros(3)))

            self.logger.debug(f"Added {len(poses_offsets)} cylinder rotations")

        # Handle prism case (ε1 → 0) - ADJUSTED FOR Y-AXIS CLOSING LINE
        elif prism_like and not cuboid_like:
            self.logger.debug(f"Detected prism shape - adding translations per original paper method (adapted for Y-axis)")

            # For prism: move along z-axis for λx and λy alignments
            # move along x,y for λz alignment
            z_vals = linspace_clamped(S.az)
            x_vals = linspace_clamped(S.ax) 
            y_vals = linspace_clamped(S.ay)
            
            # Identify which base rotations correspond to which axis alignments
            for Rg in base_R_list:
                # Check which axis the gripper closing direction (Y-axis) is most aligned with
                closing_world = Rg @ G.lambda_local  # G.lambda_local = [0, 1, 0]
                
                # Most aligned axis determines translation directions
                alignments = [abs(np.dot(closing_world, λx_w)), 
                            abs(np.dot(closing_world, λy_w)), 
                            abs(np.dot(closing_world, λz_w))]
                max_alignment_idx = np.argmax(alignments)
                
                if max_alignment_idx == 0:  # Closing line aligned with superquadric λx
                    # when λ aligned with λx, move along z-axis
                    for dz in z_vals:
                        if dz != 0:
                            Δt_world = dz * λz_w
                            poses_offsets.append((Rg, Δt_world))
                elif max_alignment_idx == 1:  # Closing line aligned with superquadric λy
                    # when λ aligned with λy, move along z-axis
                    for dz in z_vals:
                        if dz != 0:
                            Δt_world = dz * λz_w
                            poses_offsets.append((Rg, Δt_world))
                else:  # Closing line aligned with superquadric λz
                    # when λ aligned with λz, move along x,y grid on base
                    for dx in x_vals:
                        for dy in y_vals:
                            if dx != 0 or dy != 0:
                                Δt_world = dx * λx_w + dy * λy_w
                                poses_offsets.append((Rg, Δt_world))

            self.logger.debug(f"Added {len(poses_offsets)} prism translations")

        # Handle cuboid case (ε2 → 0) - ADJUSTED FOR Y-AXIS CLOSING LINE
        elif cuboid_like and not prism_like:
            self.logger.debug(f"Detected cuboid shape - adding translations per original paper method (adapted for Y-axis)")

            x_vals = linspace_clamped(S.ax)
            y_vals = linspace_clamped(S.ay)
            
            for Rg in base_R_list:
                closing_world = Rg @ G.lambda_local  # G.lambda_local = [0, 1, 0]
                
                # Check alignment with x and y axes of superquadric
                x_alignment = abs(np.dot(closing_world, λx_w))
                y_alignment = abs(np.dot(closing_world, λy_w))
                
                if x_alignment > y_alignment:  # More aligned with superquadric λx
                    # when λ aligned with λx, move along y-axis
                    for dy in y_vals:
                        if dy != 0:
                            Δt_world = dy * λy_w
                            poses_offsets.append((Rg, Δt_world))
                else:  # More aligned with superquadric λy
                    # when λ aligned with λy, move along x-axis  
                    for dx in x_vals:
                        if dx != 0:
                            Δt_world = dx * λx_w
                            poses_offsets.append((Rg, Δt_world))

            self.logger.debug(f"Added {len(poses_offsets)} cuboid translations")

        # Handle combined case (both prism and cuboid)
        elif prism_like and cuboid_like:
            self.logger.debug(f"Detected combined prism+cuboid - using general approach")
            # Keep your existing general implementation for this case
            z_vals = linspace_clamped(S.az)
            x_vals = linspace_clamped(S.ax) 
            y_vals = linspace_clamped(S.ay)

            for Rg in base_R_list:
                for dz in z_vals:
                    for dx in x_vals:
                        for dy in y_vals:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue  # Skip origin 
                            Δt_world = dx * λx_w + dy * λy_w + dz * λz_w
                            poses_offsets.append((Rg, Δt_world))

            self.logger.debug(f"Added {len(poses_offsets)} combined translations")

        return R_list_full, poses_offsets

    def make_world_pose(self, S: Superquadric, Rg, Δt=np.zeros(3)):
        """
        Place gripper closing point P_G at SQ center + Δt (world).
        Return (R_world, t_world).
        """
        return (Rg, S.T + Δt)
        
    def get_all_grasps(self, point_cloud_path, shape, scale, euler, translation):
        """
        Get ALL grasps (raw candidates) without filtering
        Returns:
            List of dictionaries with 'pose' and 'score' keys for consistency
        """
        try:
            # Create superquadric object
            S = Superquadric(ε=shape, a=scale, euler=euler, t=translation)
            G = self.gripper
            
            # Load point cloud & build KD-tree
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            if not pcd.has_points():
                return []
            X = np.asarray(pcd.points)
            kdtree = KDTree(X)
            
            # Generate raw candidates
            base_R = self.principal_axis_sweeps(S, G, step_deg=10)
            all_R, extra_offsets = self.extra_sweeps_special_shapes(S, base_R, G)

            # Build G_raw
            G_raw = []
            for Rg, Δt in extra_offsets:
                G_raw.append(self.make_world_pose(S, Rg, Δt))
            seen_rots = set()
            for Rg, _ in extra_offsets:
                key = tuple(np.round(Rg, 6).ravel())
                seen_rots.add(key)
            for Rg in all_R:
                key = tuple(np.round(Rg, 6).ravel())
                if key not in seen_rots:
                    G_raw.append(self.make_world_pose(S, Rg, np.zeros(3)))
                    seen_rots.add(key)

            self.logger.debug(f"Found {len(G_raw)} raw grasp candidates")

            # Convert to consistent dictionary format
            grasp_data = []
            for Rg, tg in G_raw:
                T = np.eye(4)
                T[:3, :3] = Rg
                T[:3, 3] = tg
                
                grasp_data.append({
                    'pose': T,
                    'score': 0.0,  # Raw candidates get default score
                    'rotation': Rg,
                    'translation': tg
                })
            
            # Store for future reference
            self.last_valid_grasps = [data['pose'] for data in grasp_data]
            
            return grasp_data  # Now returns list of dictionaries
            
        except Exception as e:
            self.logger.error(f"Error getting all grasps: {e}")
            return []

    def plan_grasps(self,
                    point_cloud_path: str,
                    shape: np.ndarray,
                    scale: np.ndarray,
                    euler: np.ndarray,
                    translation: np.ndarray,
                    *,
                    max_grasps: int = 5,
                    visualize_support_test: bool = False,
                    visualize_collision_test: bool = False):
        """
        Generate and score grasp poses around a superquadric model.

        Returns
        -------
        list[dict]  each with {'pose','score','rotation','translation'}
        """

        # ----------------------- 1. I/O -----------------------
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        if not pcd.has_points():
            self.logger.warning(f"Empty object cloud: {point_cloud_path}")
            return []

        object_pts = np.asarray(pcd.points, dtype=np.float64)        # numeric only
        kdtree     = KDTree(object_pts)                              # reused later
        self.logger.debug(f"Loaded {len(object_pts)} object points")

        # ---------------------- 2. model  ------------------------
        S = Superquadric(ε=shape, a=scale, euler=euler, t=translation)
        G = self.gripper

        # Dense surface samples for α / β metrics (≈10 k points)
        S_samples = sample_superquadric_surface(S, n_samples=10_000)

        # -------------------- 3. candidate set ------------------------------
        base_R        = self.principal_axis_sweeps(S, G, step_deg=10)     
        all_R, offset = self.extra_sweeps_special_shapes(S, base_R, G)

        G_raw = [self.make_world_pose(S, R, Δ) for R, Δ in offset]
        seen  = {tuple(np.round(R, 6).ravel()) for R, _ in offset}
        for R in all_R:
            key = tuple(np.round(R, 6).ravel())
            if key not in seen:
                G_raw.append(self.make_world_pose(S, R, np.zeros(3)))
                seen.add(key)

        # ---------------------- 4. collision filter ------------------------
        G_support = [pose for pose in G_raw
                    if support_test(*pose, S, G, kdtree=kdtree,
                                    debug_mode=visualize_support_test,
                                    max_debug_calls=3)]

        self.logger.info(f"{len(G_support)}/{len(G_raw)} grasps remain after support filtering")
        
        # ---------------------- 5. collision filter ----------------------
        G_valid   = [pose for pose in G_support
                    if collision_test(*pose, S, G, kdtree=kdtree,
                                    debug_mode=visualize_collision_test,
                                    max_debug_calls=3)]

        self.logger.info(f"{len(G_valid)}/{len(G_support)} grasps remain after collision filtering")
        if not G_valid:
            self.logger.warning("No valid grasps after filtering")
            return []
        
        # ---------------------- 6. scoring ----------------------
        scored = []
        for Rg, tg in G_valid:
            score, (hα, hβ, hγ, hδ) = score_grasp(
                Rg, tg, S_samples, object_pts, kdtree=kdtree)
            self.logger.info(f"score={score:.3f} α={hα:.3f} β={hβ:.3f} γ={hγ:.3f} δ={hδ:.3f}")
            scored.append((score, Rg, tg))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_k = scored[:max_grasps]

        # ---------------------- 7. package ----------------------
        results = []
        for score, Rg, tg in best_k:
            T         = np.eye(4)
            T[:3,:3]  = Rg
            T[:3, 3]  = tg
            results.append(dict(pose=T,
                                score=score,
                                rotation=Rg,
                                translation=tg))
        return results

    def select_best_grasp_with_criteria(self, grasp_data_list, object_center=None, point_cloud=None):
        """
        Select best grasp with criteria 
        """
        try:
            if not grasp_data_list:
                self.logger.warning("No grasp data provided")
                return None
                
            # Define constants for scoring only
            robot_origin = np.array([0.0, 0.0, 0.0])
            preferred_approach = np.array([0.0, 0.0, 1.0])  # Approach from above
            
            # All grasp_data_list entries are tested for support and collision
            
            # Score all valid grasps (no filtering, just scoring)
            best_score = -float('inf')
            best_grasp = None
            best_info = None
            
            for original_idx, grasp_data in enumerate(grasp_data_list):
                try:
                    pose = grasp_data['pose']
                    base_score = grasp_data['score']
                    position = pose[:3, 3]
                    rotation_matrix = pose[:3, :3]
                    
                    # Convert to scalars for scoring calculations
                    pos_x, pos_y, pos_z = float(position[0]), float(position[1]), float(position[2])
                    position_scalar = np.array([pos_x, pos_y, pos_z])
                    
                    gripper_z_world = rotation_matrix[:, 2]
                    actual_approach_direction = -gripper_z_world
                    approach_z = float(actual_approach_direction[2])
                    
                    # === SCORING ONLY (no filtering) ===
                    
                    # Distance and reach scoring
                    distance_to_base = np.linalg.norm(position_scalar - robot_origin)
                    distance_score = 1.0 / (1.0 + distance_to_base)
                    
                    max_reach = 0.855
                    reach_penalty = max(0.0, (distance_to_base - max_reach) * 3.0)
                    
                    # Height preference scoring (not filtering)
                    height_score = 1.0
                    if pos_z < 0.05:  # Prefer higher grasps
                        height_score = 0.3
                    elif pos_z > 0.2:
                        height_score = 0.7
                    
                    # Approach direction preference (not filtering)
                    approach_alignment = np.dot(actual_approach_direction, preferred_approach)
                    approach_score = max(0.0, approach_alignment)
                    
                    # Strong penalty for approaches from below table
                    below_table_penalty = 0.0
                    if approach_z < -0.5:  # Approaching from significantly below
                        below_table_penalty = 5.0 * abs(approach_z)  # Heavy penalty
                    elif approach_z < 0.0:  # Any downward approach
                        below_table_penalty = 2.0 * abs(approach_z)  # Moderate penalty
                    
                    # Point cloud distance scoring (if available)
                    point_cloud_distance_score = 0.0
                    point_cloud_penalty = 0.0
                    min_distance_to_cloud = None
                    
                    if point_cloud is not None:
                        try:
                            distances = np.linalg.norm(point_cloud - position_scalar, axis=1)
                            min_distance_to_cloud = np.min(distances)
                            
                            # Preference for grasps near object surface (not filtering)
                            if 0.01 <= min_distance_to_cloud <= 0.03:  # Sweet spot
                                point_cloud_distance_score = 1.0
                            elif min_distance_to_cloud > 0.03:
                                point_cloud_distance_score = 0.5
                            else:  # Very close < 1cm
                                point_cloud_distance_score = 0.2
                                
                        except Exception as pc_error:
                            self.logger.warning(f"Point cloud distance calculation failed: {pc_error}")
                    
                    # Workspace preference scoring
                    workspace_score = 0.0
                    if 0.2 <= pos_x <= 0.6 and -0.3 <= pos_y <= 0.3:
                        workspace_score = 1.0
                    elif 0.1 <= pos_x <= 0.7 and -0.4 <= pos_y <= 0.4:
                        workspace_score = 0.5
                    
                    # FINAL SCORE CALCULATION
                    final_score = (
                        base_score +                           # Original grasp quality
                        2.0 * distance_score +                # Prefer closer grasps
                        1.5 * height_score +                  # Height preference
                        2.0 * approach_score +                # Approach direction preference
                        1.0 * workspace_score +               # Workspace preference
                        1.0 * point_cloud_distance_score +    # Object proximity preference
                        -reach_penalty                        # Penalties
                        -below_table_penalty
                    )
                    
                    # Debug output for first few grasps
                    if original_idx < 3:
                        self.logger.info(f"  [SCORE] Grasp {original_idx+1}:")
                        self.logger.info(f"    Position: [{pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f}]")
                        self.logger.info(f"    Approach Z: {approach_z:.3f}")
                        self.logger.info(f"    Distance to base: {distance_to_base:.3f}m")
                        self.logger.info(f"    Final score: {final_score:.6f}")

                    # Track best grasp
                    if final_score > best_score:
                        best_score = final_score
                        best_grasp = grasp_data
                        best_info = {
                            'original_index': original_idx,
                            'base_score': base_score,
                            'distance_score': distance_score,
                            'height_score': height_score,
                            'approach_score': approach_score,
                            'workspace_score': workspace_score,
                            'point_cloud_distance_score': point_cloud_distance_score,
                            'min_distance_to_cloud': min_distance_to_cloud,
                            'final_score': final_score
                        }
                    
                except Exception as e:
                    self.logger.error(f"Error scoring grasp {original_idx+1}: {e}")
                    continue
            
            if best_grasp is not None:
                self.logger.info(f"\n[SELECTION] Best grasp selected (#{best_info['original_index']+1}) with score: {best_score:.6f}")
                self.logger.info(f"   Base score: {best_info['base_score']:.6f}")
                self.logger.info(f"   Distance score: {best_info['distance_score']:.3f}")
                self.logger.info(f"   Height score: {best_info['height_score']:.3f}")
                self.logger.info(f"   Approach score: {best_info['approach_score']:.3f}")

                if best_info.get('min_distance_to_cloud') is not None:
                    self.logger.info(f"   Min distance to point cloud: {best_info['min_distance_to_cloud']*1000:.1f}mm")

                # Show final pose
                best_pose = best_grasp['pose']
                pos = best_pose[:3, 3]
                self.logger.info(f"   FINAL POSITION: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

            return best_grasp
            
        except Exception as e:
            self.logger.error(f"Error in select_best_grasp_with_criteria: {e}")
            return None
        
    def _get_gripper_collision_points(self, gripper_center, rotation_matrix):
        """
        Get key points on the gripper that could collide with the object
        
        Args:
            gripper_center: Gripper center position in world frame
            rotation_matrix: Gripper orientation (3x3 rotation matrix)
            
        Returns:
            List of 3D points representing gripper collision zones
        """
        points = []
        
        # Get gripper axes in world frame
        approach_world = rotation_matrix @ self.gripper.approach_axis  # -Z local
        closing_world = rotation_matrix @ self.gripper.lambda_local    # +Y local  
        width_world = rotation_matrix @ np.array([1, 0, 0])           # +X local
        
        # Finger tip positions
        half_open = self.gripper.max_open / 2.0
        tip1 = gripper_center + closing_world * half_open
        tip2 = gripper_center - closing_world * half_open
        points.extend([tip1, tip2])
        
        # Finger body points (along finger length)
        finger_steps = 3  # Check 3 points along each finger
        for i in range(1, finger_steps + 1):
            finger_offset = (-approach_world) * (i / finger_steps) * self.gripper.jaw_len
            points.append(tip1 + finger_offset)
            points.append(tip2 + finger_offset)
        
        # Palm/back points
        palm_offset = (-approach_world) * (self.gripper.jaw_len + self.gripper.palm_depth/2)
        points.append(gripper_center + palm_offset)
        
        # Side points (gripper width)
        for side_factor in [-0.5, 0.5]:
            side_offset = width_world * side_factor * self.gripper.palm_width
            points.append(gripper_center + palm_offset + side_offset)
        
        # Cross-bar points (connecting the fingers)
        crossbar_offset = (-approach_world) * self.gripper.jaw_len
        for closing_factor in [-0.8, -0.4, 0.0, 0.4, 0.8]:
            crossbar_point = gripper_center + crossbar_offset + closing_world * closing_factor * self.gripper.max_open
            points.append(crossbar_point)
        
        return points
    