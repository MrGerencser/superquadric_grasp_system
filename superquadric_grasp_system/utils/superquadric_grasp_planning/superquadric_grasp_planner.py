import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R_simple
from ..grasp_selector import select_best_grasp


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

        self.logger.debug(f"This implementation assumes gripper closing line λ is along Y-axis (not X-axis as in original)")
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
        Add extra grasps based on superquadric shape:
        1) ε1 → 0 (prism): slide gripper along z-axis in 15mm steps
        2) ε2 → 0 (rectangular cross-section): slide along perpendicular axis in 15mm steps  
        3) ε2 = 1 and ax = ay (circular cross-section): rotate around z-axis in π/8 steps
        """
        R_list_full = list(base_R_list)
        poses_offsets = []

        # Thresholds
        prism_like = (S.ε1 < 0.1)          # ε1 → 0 (more restrictive)
        rectangular_like = (S.ε2 < 0.1)    # ε2 → 0 (more restrictive)
        circular_like = (abs(S.ε2 - 1.0) < 0.05) and (abs(S.ax - S.ay) < 0.005)  # ε2 = 1, ax = ay

        self.logger.debug(f"Shape analysis: ε1={S.ε1:.3f}, ε2={S.ε2:.3f}, ax={S.ax:.3f}, ay={S.ay:.3f}")
        self.logger.debug(f"prism_like={prism_like}, rectangular_like={rectangular_like}, circular_like={circular_like}")

        # Get SQ-local axes in world coordinates
        λx_w, λy_w, λz_w = S.axes_world[:, 0], S.axes_world[:, 1], S.axes_world[:, 2]

        # 15mm step size
        step = 0.015

        def get_limited_steps(extent, max_steps=5):
            """Limit the number of steps to prevent explosion of candidates"""
            if extent < 1e-6:
                return np.array([0.0])
            n = min(int(np.floor(extent / step)), max_steps)  # Limit to max_steps
            if n == 0:
                return np.array([0.0])
            return np.arange(-n, n+1) * step

        # CASE 1: Cylinder (most specific, check first)
        if prism_like and circular_like:
            self.logger.debug("Detected CYLINDER: adding π/8 rotations around z-axis")
            
            # Only add rotations around z-axis, no translations
            axis_world = λz_w
            R_align = rotation_from_u_to_v(G.lambda_local, axis_world)

            # π/8 = 22.5 degrees
            angle_step = np.pi / 8  # 22.5 degrees
            for i in range(1, 16):  # Skip 0 (already in base set), add 15 more = 16 total
                extra_angle = i * angle_step
                R_extra = R_simple.from_rotvec(axis_world * extra_angle).as_matrix()
                R_new = R_extra @ R_align
                R_list_full.append(R_new)
                poses_offsets.append((R_new, np.zeros(3)))  # No translation offset

            self.logger.debug(f"Added {15} cylinder rotations")

        # CASE 2: Prism (ε1 → 0, but not cylinder)
        elif prism_like and not circular_like:
            self.logger.debug("Detected PRISM: adding z-axis translations for λx/λy, xy-grid for λz")
            
            z_steps = get_limited_steps(S.az)
            x_steps = get_limited_steps(S.ax)
            y_steps = get_limited_steps(S.ay)
            
            for Rg in base_R_list:
                closing_world = Rg @ G.lambda_local
                
                # Determine which axis the gripper is most aligned with
                alignments = [abs(np.dot(closing_world, λx_w)), 
                            abs(np.dot(closing_world, λy_w)), 
                            abs(np.dot(closing_world, λz_w))]
                max_alignment_idx = np.argmax(alignments)
                
                if max_alignment_idx in [0, 1]:  # Aligned with λx or λy
                    # Move along z-axis only
                    for dz in z_steps:
                        if abs(dz) > 1e-6:  # Skip zero offset
                            Δt_world = dz * λz_w
                            poses_offsets.append((Rg, Δt_world))
                            
                else:  # Aligned with λz
                    # Move on xy-grid
                    for dx in x_steps:
                        for dy in y_steps:
                            if abs(dx) > 1e-6 or abs(dy) > 1e-6:  # Skip zero offset
                                Δt_world = dx * λx_w + dy * λy_w
                                poses_offsets.append((Rg, Δt_world))

            self.logger.debug(f"Added {len(poses_offsets)} prism translations")

        # CASE 3: Rectangular cross-section (ε2 → 0, but not prism)
        elif rectangular_like and not prism_like:
            self.logger.debug("Detected RECTANGULAR: adding perpendicular translations")
            
            x_steps = get_limited_steps(S.ax)
            y_steps = get_limited_steps(S.ay)
            
            for Rg in base_R_list:
                closing_world = Rg @ G.lambda_local
                
                # Check alignment with x and y axes only
                x_alignment = abs(np.dot(closing_world, λx_w))
                y_alignment = abs(np.dot(closing_world, λy_w))
                
                if x_alignment > y_alignment:  # More aligned with λx
                    # Move along y-axis (perpendicular)
                    for dy in y_steps:
                        if abs(dy) > 1e-6:
                            Δt_world = dy * λy_w
                            poses_offsets.append((Rg, Δt_world))
                else:  # More aligned with λy
                    # Move along x-axis (perpendicular)
                    for dx in x_steps:
                        if abs(dx) > 1e-6:
                            Δt_world = dx * λx_w
                            poses_offsets.append((Rg, Δt_world))

            self.logger.debug(f"Added {len(poses_offsets)} rectangular translations")

        # CASE 4: Combined prism + rectangular (cuboid)
        elif prism_like and rectangular_like and not circular_like:
            self.logger.debug("Detected CUBOID: using combined approach with limits")
            
            # Very limited grid to prevent explosion
            z_steps = get_limited_steps(S.az, max_steps=2)  # Max ±2 steps
            x_steps = get_limited_steps(S.ax, max_steps=2)
            y_steps = get_limited_steps(S.ay, max_steps=2)

            for Rg in base_R_list:
                for dz in z_steps:
                    for dx in x_steps:
                        for dy in y_steps:
                            if abs(dx) > 1e-6 or abs(dy) > 1e-6 or abs(dz) > 1e-6:
                                Δt_world = dx * λx_w + dy * λy_w + dz * λz_w
                                poses_offsets.append((Rg, Δt_world))
                                
                                # Hard limit to prevent explosion
                                if len(poses_offsets) > 100:
                                    break
                        if len(poses_offsets) > 100:
                            break
                    if len(poses_offsets) > 100:
                        break

            self.logger.debug(f"Added {len(poses_offsets)} cuboid translations (limited)")

        else:
            self.logger.debug("Standard shape: no extra sweeps needed")

        # Log final count
        total_candidates = len(R_list_full) + len(poses_offsets)
        self.logger.debug(f"Extra sweeps complete: {len(poses_offsets)} offset poses, {total_candidates} total candidates")

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

    def plan_filtered_grasps(self, point_cloud_path, shape, scale, euler, translation,
                        visualize_support_test=False, visualize_collision_test=False):
        """Plan grasps with support and collision filtering - returns ALL valid grasps"""

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
            closing_dir_world = Rg @ G.lambda_local  # Gripper closing dir in world frame
            closing_dir_local = S.R.T @ closing_dir_world  # Transform to SQ local frame
            closing_dir_local = closing_dir_local / (np.linalg.norm(closing_dir_local) + 1e-15)  # Normalize
            
            score, (hα, hβ, hγ, hδ) = score_grasp(
                Rg, tg, S, object_pts, closing_dir_local, kdtree=kdtree)
            self.logger.info(f"score={score:.3f} α={hα:.3f} β={hβ:.3f} γ={hγ:.3f} δ={hδ:.3f}")
            scored.append((score, Rg, tg))

        scored.sort(key=lambda x: x[0], reverse=True)

        # ---------------------- 7. package all valid grasps ----------------------
        results = []
        for score, Rg, tg in scored:
            T         = np.eye(4)
            T[:3,:3]  = Rg
            T[:3, 3]  = tg
            results.append(dict(pose=T,
                                score=score,
                                rotation=Rg,
                                translation=tg))
            
        return results

    def select_best_grasp_with_criteria(self, grasp_data_list, point_cloud=None):
        """Select best grasp using unified criteria"""
        try:
            if not grasp_data_list:
                self.logger.warning("No grasp data provided")
                return None
            
            # For superquadric: filter to best-scoring grasps first
            sorted_grasps = sorted(grasp_data_list, key=lambda x: x.get('score', 0.0), reverse=True)
            best_base_score = sorted_grasps[0].get('score', 0.0)
            
            # Only evaluate grasps with the same top score
            candidates = [g for g in sorted_grasps if g.get('score', 0.0) == best_base_score]
            
            self.logger.info(f"Evaluating {len(candidates)}/{len(grasp_data_list)} grasps "
                            f"with best score: {best_base_score:.6f}")
            
            return select_best_grasp(
                grasp_poses=candidates,
                point_cloud=point_cloud,
                workspace_bounds=None,  # Use defaults
                logger=self.logger,
            )
            
        except Exception as e:
            self.logger.error(f"Error in select_best_grasp_with_criteria: {e}")
            return None