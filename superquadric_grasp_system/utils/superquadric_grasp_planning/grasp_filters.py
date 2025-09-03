from matplotlib.pylab import gamma
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from ...visualization.main_visualizer import PerceptionVisualizer

def support_test(R, t, S, G, kdtree, κ=12, r_support=None, h_support=0.02, 
                debug_mode=False, max_debug_calls=5):
    """
    True cylinder support test with optional visualization:
    - r_support: cylinder radius (defaults to half jaw width)
    - h_support: half cylinder height
    """
    # Closing direction in world
    closing_dir = R @ G.lambda_local
    closing_dir = closing_dir / np.linalg.norm(closing_dir)

    # Default support radius: half the jaw opening
    if r_support is None:
        r_support = 3 * 0.003      # e.g. 3 × voxel_size (≈9 mm)
    if h_support is None:
        h_support = G.jaw_len      # full finger length

    # two finger-contact points on the SQ surface, along closing_dir
    tip1 = t + closing_dir * h_support
    tip2 = t - closing_dir * h_support

    # Get all points
    X = kdtree.data
    
    # Compute point projections onto closing axis for tip1
    rel1 = X - tip1
    proj1 = np.dot(rel1, closing_dir)
    radial1 = np.linalg.norm(rel1 - np.outer(proj1, closing_dir), axis=1)
    mask1 = (np.abs(proj1) <= h_support) & (radial1 <= r_support)
    cnt1 = np.count_nonzero(mask1)

    # Compute point projections onto closing axis for tip2
    rel2 = X - tip2
    proj2 = np.dot(rel2, closing_dir)
    radial2 = np.linalg.norm(rel2 - np.outer(proj2, closing_dir), axis=1)
    mask2 = (np.abs(proj2) <= h_support) & (radial2 <= r_support)
    cnt2 = np.count_nonzero(mask2)

    support_result = (cnt1 >= κ) and (cnt2 >= κ)

    # --- DEBUG VISUALIZATION ---
    if debug_mode:
        if not hasattr(support_test, 'debug_call_count'):
            support_test.debug_call_count = 0
        
        support_test.debug_call_count += 1
        
        if support_test.debug_call_count <= max_debug_calls:
            try:
                print(f"\n{'='*60}")
                print(f"[SUPPORT DEBUG #{support_test.debug_call_count}]")
                print(f"  Closing direction: {closing_dir}")
                print(f"  Grasp center: {t}")
                print(f"  Tip1 center: {tip1}")
                print(f"  Tip2 center: {tip2}")
                print(f"  Support cylinder radius: {r_support:.4f}m")
                print(f"  Support cylinder half-height: {h_support:.4f}m")
                print(f"  Required points per tip: {κ}")
                print(f"  Tip1 support points: {cnt1}/{len(X)}")
                print(f"  Tip2 support points: {cnt2}/{len(X)}")
                print(f"  Support test result: {'PASS' if support_result else 'FAIL'}")
                
                # Create proper 4x4 transformation matrix for visualization
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                
                # Prepare detailed support test data
                support_test_data = {
                    'R_world': R,
                    't_world': t,
                    'S': S,
                    'G': G,
                    'closing_dir': closing_dir,
                    'tip1': tip1,
                    'tip2': tip2,
                    'r_support': r_support,
                    'h_support': h_support,
                    'mask1': mask1,
                    'mask2': mask2,
                    'cnt1': cnt1,
                    'cnt2': cnt2,
                    'required_points': κ,
                    'support_result': support_result
                }
                
                visualizer = PerceptionVisualizer()
                visualizer.visualize_support_test(
                    point_cloud=kdtree.data,
                    grasp_pose=T,
                    support_test_data=support_test_data,
                    window_name=f"Support Test #{support_test.debug_call_count}"
                )
                
            except Exception as viz_error:
                print(f"    [ERROR] Support visualization failed: {viz_error}")

    return support_result


def collision_test(R_world, t_world, S, G, kdtree, debug_mode=False, max_debug_calls=5):
    """
    1. Small slab (cylinder region): for safe grasping area
    2. Large slab (includes fingers): for collision detection
    
    Returns:
        True  → NO collision (pose is valid)
        False → collision detected (reject this pose)
    """

    # --- 1. closing line in world coordinates -----------------------
    λ_dir = R_world @ G.lambda_local
    λ_dir /= np.linalg.norm(λ_dir)

    # --- 2. pick the SQ semi-axis most aligned with λ --------------
    λ_local = S.R.T @ λ_dir
    axis_idx = np.argmax(np.abs(λ_local))
    a_axis = [S.ax, S.ay, S.az][axis_idx]

    # --- 3. TWO DIFFERENT SLAB DEFINITIONS -------------------------
    half_open = G.max_open / 2.0
    
    # SMALL SLAB: Just the cylinder region (for safe grasping)
    half_height_cylinder = half_open
    radius = G.jaw_len
    
    # LARGE SLAB: Includes finger extent (for collision detection)
    half_height_finger = half_open + 2 * G.thickness  # Add finger length to slab
    
    # --- 4. GET POINTS FOR BOTH SLABS ------------------------------
    X = kdtree.data
    rel = X - t_world
    proj = rel @ λ_dir
    
    # Small slab mask (cylinder region only)
    small_slab_mask = np.abs(proj) <= half_height_cylinder
    
    # Large slab mask (includes finger extent)
    large_slab_mask = np.abs(proj) <= half_height_finger
    
    if not np.any(large_slab_mask):
        collision_result = True  # No points in either slab → no collision
        if debug_mode:
            print(f"    [COLLISION] NO POINTS IN LARGE SLAB - Safe grasp")
        return collision_result

    # --- 5. CYLINDER LOGIC: Use SMALL slab for safe grasping ------
    if np.any(small_slab_mask):
        # Points inside small slab
        small_slab_points = X[small_slab_mask]
        rel_small_slab = rel[small_slab_mask]
        proj_small_slab = proj[small_slab_mask]
        
        # Calculate radial distance from closing line for points in small slab
        radial_vec_small = rel_small_slab - np.outer(proj_small_slab, λ_dir)
        radial_len_small = np.linalg.norm(radial_vec_small, axis=1)
        
        # Points OUTSIDE cylinder in small slab = potential collision points
        outside_cylinder_mask_small = radial_len_small > radius
        potential_collision_points_from_cylinder = small_slab_points[outside_cylinder_mask_small]
    else:
        potential_collision_points_from_cylinder = np.array([]).reshape(0, 3)
        outside_cylinder_mask_small = np.array([], dtype=bool)

    # --- 6. FINGER COLLISION: Use LARGE slab ----------------------
    # All points in large slab that are NOT in small slab = finger collision candidates
    large_slab_points = X[large_slab_mask]
    
    # Find points that are in large slab but NOT in small slab
    # These are the points in the finger extension region
    finger_region_mask = large_slab_mask.copy()
    finger_region_mask[small_slab_mask] = False  # Remove small slab points
    finger_region_points = X[finger_region_mask]

    # Combine all potential collision points
    if len(potential_collision_points_from_cylinder) > 0 and len(finger_region_points) > 0:
        all_potential_collision_points = np.vstack([
            potential_collision_points_from_cylinder,
            finger_region_points
        ])
    elif len(potential_collision_points_from_cylinder) > 0:
        all_potential_collision_points = potential_collision_points_from_cylinder
    elif len(finger_region_points) > 0:
        all_potential_collision_points = finger_region_points
    else:
        collision_result = True  # No potential collision points → safe
        if debug_mode:
            print(f"    [COLLISION] No potential collision points - Safe grasp")
        return collision_result

    # --- 7. CHECK GRIPPER BODY COLLISIONS -------------------------
    # Check collision with gripper palm/back
    palm_collisions = check_palm_collision(
        all_potential_collision_points, t_world, R_world, G
    )
    
    # Check collision with gripper fingers
    finger_collisions = check_finger_collision(
        all_potential_collision_points, t_world, R_world, G
    )
    
    # Combine both collision types
    has_palm_collision = np.any(palm_collisions) if len(palm_collisions) > 0 else False
    has_finger_collision = np.any(finger_collisions) if len(finger_collisions) > 0 else False
    has_collision = has_palm_collision or has_finger_collision
    
    collision_result = not has_collision  # True = no collision

    # --- 8. DEBUG OUTPUT -------------------------------------------
    if debug_mode:
        if not hasattr(collision_test, 'debug_call_count'):
            collision_test.debug_call_count = 0
        
        collision_test.debug_call_count += 1
        
        if collision_test.debug_call_count <= max_debug_calls:
            try:
                small_slab_count = np.sum(small_slab_mask)
                large_slab_count = np.sum(large_slab_mask)
                finger_region_count = len(finger_region_points)
                cylinder_collision_count = len(potential_collision_points_from_cylinder)
                
                print(f"\n{'='*60}")
                print(f"[COLLISION DEBUG #{collision_test.debug_call_count}] - TWO-SLAB LOGIC")
                print(f"  Small slab (cylinder): half_height={half_height_cylinder:.4f}m, radius={radius:.4f}m")
                print(f"  Large slab (w/fingers): half_height={half_height_finger:.4f}m")
                print(f"  Closing direction λ: {λ_dir}")
                print(f"  Grasp center: {t_world}")
                print(f"  Points in small slab: {small_slab_count}/{len(X)}")
                print(f"  Points in large slab: {large_slab_count}/{len(X)}")
                print(f"  Points in finger region: {finger_region_count}")
                print(f"  Cylinder collision candidates: {cylinder_collision_count}")
                print(f"  Total collision candidates: {len(all_potential_collision_points)}")
                print(f"  Palm collisions: {np.sum(palm_collisions) if len(palm_collisions) > 0 else 0}")
                print(f"  Finger collisions: {np.sum(finger_collisions) if len(finger_collisions) > 0 else 0}")
                print(f"  Total collisions: {has_collision}")
                print(f"  Result: {'NO COLLISION' if collision_result else 'COLLISION DETECTED'}")
                
                # Create proper 4x4 transformation matrix for visualization
                T = np.eye(4)
                T[:3, :3] = R_world
                T[:3, 3] = t_world
                
                # Prepare detailed collision test data
                collision_test_data = {
                    'R_world': R_world,
                    't_world': t_world,
                    'S': S,
                    'G': G,
                    'lambda_dir': λ_dir,
                    'radius': radius,
                    'half_height_cylinder': half_height_cylinder,
                    'half_height_finger': half_height_finger,
                    'small_slab_mask': small_slab_mask,
                    'large_slab_mask': large_slab_mask,
                    'outside_cylinder_mask_small': outside_cylinder_mask_small,
                    'finger_region_mask': finger_region_mask,
                    'palm_collisions': palm_collisions,
                    'finger_collisions': finger_collisions,
                    'all_potential_collision_points': all_potential_collision_points,
                    'collision_result': collision_result
                }
                
                visualizer = PerceptionVisualizer()
                visualizer.visualize_collision_test(
                    point_cloud=kdtree.data,
                    grasp_pose=T,
                    collision_test_data=collision_test_data,
                    window_name=f"Collision Test #{collision_test.debug_call_count}"
                )
                
            except Exception as viz_error:
                print(f"    [ERROR] Visualization failed: {viz_error}")

    return collision_result

def check_finger_collision(points, gripper_center, R_world, gripper):
    """Check collision with gripper fingers modeled as cylinders in world frame"""
    if len(points) == 0:
        return np.array([], dtype=bool)
    
    # Work entirely in world coordinates
    rel_points = points - gripper_center
    
    # Get gripper axes in world frame
    approach_world = R_world @ gripper.approach_axis  # -Z in local becomes approach direction in world
    closing_world = R_world @ gripper.lambda_local    # +Y in local becomes closing direction in world
    width_world = R_world @ np.array([1, 0, 0])       # +X in local becomes width direction in world
    
    # Project points onto gripper axes
    approach_proj = rel_points @ approach_world
    closing_proj = rel_points @ closing_world
    width_proj = rel_points @ width_world
    
    # Finger cylinder parameters
    finger_radius = gripper.thickness
    finger_half_width = gripper.max_open / 2.0
    
    # Fingers extend TOWARD the object (in POSITIVE approach direction)
    # Since approach_axis = [0, 0, -1] and gripper moves toward object in -Z direction,
    # the approach_world vector points toward the object
    # Fingers extend from gripper_center in the approach_world direction
    in_finger_length = (approach_proj >= 0) & (approach_proj <= gripper.jaw_len)
    
    # Calculate distance from each finger's center line
    # Left finger center line: closing_proj = +finger_half_width, any width_proj, approach_proj ∈ [0, +jaw_len]
    left_finger_closing_dist = np.abs(closing_proj - finger_half_width)
    left_finger_width_dist = np.abs(width_proj)
    left_finger_radial_dist = np.sqrt(left_finger_closing_dist**2 + left_finger_width_dist**2)
    
    # Right finger center line: closing_proj = -finger_half_width, any width_proj, approach_proj ∈ [0, +jaw_len]  
    right_finger_closing_dist = np.abs(closing_proj + finger_half_width)
    right_finger_width_dist = np.abs(width_proj)
    right_finger_radial_dist = np.sqrt(right_finger_closing_dist**2 + right_finger_width_dist**2)
    
    # Check if points are within cylinder radius of either finger
    near_left_finger = left_finger_radial_dist <= finger_radius
    near_right_finger = right_finger_radial_dist <= finger_radius
    
    # Collision occurs if point is in finger length AND near either finger cylinder
    finger_collisions = in_finger_length & (near_left_finger | near_right_finger)
    
    return finger_collisions

def check_palm_collision(points, gripper_center, R_world, gripper):
    """Check collision in world frame directly"""
    if len(points) == 0:
        return np.array([], dtype=bool)
    
    # Work entirely in world coordinates
    rel_points = points - gripper_center
    
    # Get gripper axes in world frame
    approach_world = R_world @ gripper.approach_axis  # -Z in local becomes approach direction in world
    closing_world = R_world @ gripper.lambda_local    # +Y in local becomes closing direction in world
    width_world = R_world @ np.array([1, 0, 0])       # +X in local becomes width direction in world
    
    # Project points onto gripper axes
    approach_proj = rel_points @ approach_world
    closing_proj = rel_points @ closing_world
    width_proj = rel_points @ width_world
    
    # Palm is in the approach direction (where gripper comes from)
    # Gripper approaches from positive approach direction, so palm is at positive approach_proj
    in_approach_region = (approach_proj >= gripper.jaw_len) & (approach_proj <= (gripper.jaw_len + gripper.palm_depth))
    within_palm_width = np.abs(width_proj) <= gripper.palm_width / 2
    within_palm_height = np.abs(closing_proj) <= gripper.max_open / 2
    
    collisions = in_approach_region & within_palm_width & within_palm_height
    return collisions

def score_grasp(R, t, S, Y, closing_dir_local, *,
                d_th=0.05,                  # coverage distance threshold
                q_alpha=0.002,              # goodness-of-fit decay
                q_gamma=0.5,                # curvature-flatness decay
                q_delta=0.005,              # COM-distance decay
                n_surface_samples=10000,
                kdtree=None):  
    """
    Evaluate one grasp candidate.

    Parameters
    ----------
    R, t : ndarray
        3×3 rotation and 3-vector translation of the gripper in the object frame.
    S : (M,3) ndarray
        Dense, uniform samples on the recovered superquadric surface.
    Y : (N,3) ndarray
        Observed Object points associated with that superquadric.
    ...

    Returns
    -------
    Overall score h = hα·hβ·hγ·hδ
    Individual components (hα, hβ, hγ, hδ).
    """
    # Sample surface for coverage/goodness calculations
    S_samples = sample_superquadric_surface(S, n_surface_samples)

    # ---------- Goodness of fit (hα) ----------
    if kdtree is None:
        kdtree = KDTree(Y)
    d2, _ = kdtree.query(S_samples, k=1)
    alpha = d2.mean()
    h_alpha = np.exp(-(alpha**2) / q_alpha)

    # ---------- Coverage (hβ) ----------
    covered = (d2 <= d_th**2)
    beta = covered.sum() / S_samples.shape[0]
    h_beta = beta**2
    
    # ---- Curvature flatness (hγ) ----
    gamma = endpoint_curvature(S, closing_dir_local)
    print("gamma:", gamma)
    # h_gamma = np.exp(-(gamma**2) / q_gamma) # original
    h_gamma = 1.0 / (1.0 + gamma / q_gamma)  # less aggressive decay

    # ---------- COM distance (hδ) ----------
    com   = Y.mean(axis=0)
    pg    = t
    delta = np.linalg.norm(pg - com)
    h_delta = np.exp(-(delta**2) / q_delta)

    # ---------- Overall score ----------
    return h_alpha * h_beta * h_gamma * h_delta, (h_alpha, h_beta, h_gamma, h_delta)

def endpoint_curvature(S, closing_dir_local):
    """
    Compute Gaussian curvature at superquadric axis endpoints.
    
    Parameters:
    - S: Superquadric object with ax, ay, az, ε1, ε2
    - closing_dir_local: Grasp direction in superquadric local frame
    
    Returns:
    - gamma: Average absolute Gaussian curvature at contact points
    """
    a, b, c = S.ax, S.ay, S.az
    e1, e2 = S.ε1, S.ε2
    
    # Determine which axis we're grasping along
    abs_dir = np.abs(closing_dir_local)
    axis_idx = np.argmax(abs_dir)
    
    # Handle degenerate cases
    if abs(e1) < 1e-12 or abs(e2) < 1e-12:
        return 10.0  # Very sharp edge
    
    # Prevent division by zero
    e1_safe = max(abs(e1), 1e-12)
    e2_safe = max(abs(e2), 1e-12)
    
    if axis_idx == 0:  # x-axis grasp, contact at (±a, 0, 0)
        # Principal curvatures in y and z directions
        k1 = (2.0/e2_safe - 1.0) / (b * e2_safe)  # y-direction (xy-plane)
        k2 = (2.0/e1_safe - 1.0) / (c * e1_safe)  # z-direction (principal axis)
        
    elif axis_idx == 1:  # y-axis grasp, contact at (0, ±b, 0)
        # Principal curvatures in x and z directions  
        k1 = (2.0/e2_safe - 1.0) / (a * e2_safe)  # x-direction (xy-plane)
        k2 = (2.0/e1_safe - 1.0) / (c * e1_safe)  # z-direction (principal axis)
        
    else:  # z-axis grasp, contact at (0, 0, ±c) - CORRECTED
        # Both principal curvatures are in the xy-plane, so both use ε2
        k1 = (2.0/e2_safe - 1.0) / (a * e2_safe)  # x-direction (xy-plane)
        k2 = (2.0/e2_safe - 1.0) / (b * e2_safe)  # y-direction (xy-plane)
    
    gamma = abs(k1 * k2)
    
    # Scale to reasonable range for grasp planning
    gamma = gamma * 0.005
    
    return gamma


def sample_superquadric_surface(S, n_samples=1000):
    """
    Sample points densely on superquadric surface for α computation.
    """
    # Create parameter grid
    u = np.linspace(-np.pi/2, np.pi/2, int(np.sqrt(n_samples)))
    v = np.linspace(-np.pi, np.pi, int(np.sqrt(n_samples)))
    U, V = np.meshgrid(u, v)
    U, V = U.flatten(), V.flatten()
    
    # Parametric surface equations
    eps1, eps2 = S.ε1, S.ε2
    x = S.ax * np.sign(np.cos(U)) * (np.abs(np.cos(U))**eps1) * np.sign(np.cos(V)) * (np.abs(np.cos(V))**eps2)
    y = S.ay * np.sign(np.cos(U)) * (np.abs(np.cos(U))**eps1) * np.sign(np.sin(V)) * (np.abs(np.sin(V))**eps2)
    z = S.az * np.sign(np.sin(U)) * (np.abs(np.sin(U))**eps1)
    
    # Transform to world coordinates
    points_local = np.column_stack([x, y, z])
    points_world = (S.R @ points_local.T).T + S.T
    
    return points_world
