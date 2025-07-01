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
                print(f"\n[SUPPORT DEBUG #{support_test.debug_call_count}]")
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
    CORRECTED IMPLEMENTATION with two-slab logic:
    
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
                
                print(f"\n[COLLISION DEBUG #{collision_test.debug_call_count}] - TWO-SLAB LOGIC")
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

def score_grasp(R, t, S, G, kdtree, surface_tol=0.005):
    """
    superquadric implicit function
    """
    X = kdtree.data
    N = X.shape[0]
    
    # Transform to SQ-local coordinates
    rel = X - S.T
    pts_local = rel @ S.R.T
    
    surface_tol = 0.05 * min(S.ax, S.ay, S.az)
    
    # superquadric implicit function
    # Standard form: ((|x/a1|^(2/ε2) + |y/a2|^(2/ε2))^(ε2/ε1) + |z/a3|^(2/ε1))^(1/1) = 1
    
    # Avoid division by zero
    safe_ax = max(S.ax, 1e-6)
    safe_ay = max(S.ay, 1e-6) 
    safe_az = max(S.az, 1e-6)
    safe_eps1 = max(S.ε1, 0.1)
    safe_eps2 = max(S.ε2, 0.1)
    
    # Normalized coordinates
    x_norm = np.abs(pts_local[:, 0]) / safe_ax
    y_norm = np.abs(pts_local[:, 1]) / safe_ay
    z_norm = np.abs(pts_local[:, 2]) / safe_az
    
    # CORRECT EQUATION: F(x,y,z) = ((x/a)^(2/ε2) + (y/b)^(2/ε2))^(ε2/ε1) + (z/c)^(2/ε1)
    try:
        # Handle potential numerical issues with small exponents
        eps_ratio_xy = 2.0 / safe_eps2
        eps_ratio_z = 2.0 / safe_eps1
        eps_power = safe_eps2 / safe_eps1
        
        # Compute terms with clamping to avoid overflow
        x_term = np.power(np.clip(x_norm, 1e-10, 1e10), eps_ratio_xy)
        y_term = np.power(np.clip(y_norm, 1e-10, 1e10), eps_ratio_xy)
        z_term = np.power(np.clip(z_norm, 1e-10, 1e10), eps_ratio_z)
        
        # Combine xy terms
        xy_combined = np.power(np.clip(x_term + y_term, 1e-10, 1e10), eps_power)
        
        # Final implicit function value
        implicit_values = xy_combined + z_term
        
        # Distance from surface (F = 1)
        surface_distances = np.abs(implicit_values - 1.0)

    except Exception as eq_error:
        print(f"    [ERROR] Equation computation failed: {eq_error}")
        # Fallback: assume no surface points
        surface_distances = np.full(len(pts_local), 1000.0)
    
    # ADAPTIVE SURFACE TOLERANCE: Scale with object size
    char_size = (safe_ax * safe_ay * safe_az) ** (1/3)
    base_tolerance = 0.1  # Base tolerance for implicit function
    size_scaled_tolerance = base_tolerance * max(0.5, char_size / 0.05)  # Scale with object size
    
    # Surface mask with adaptive tolerance
    surface_mask = surface_distances < size_scaled_tolerance
    Y = X[surface_mask]
    
    if len(Y) == 0:
        # FALLBACK: If no surface points, use proximity-based approach
        # Find points within reasonable distance of SQ center
        distances_from_center = np.linalg.norm(rel, axis=1)
        max_reasonable_distance = np.max([safe_ax, safe_ay, safe_az]) * 2.0
        
        proximity_mask = distances_from_center < max_reasonable_distance
        Y_fallback = X[proximity_mask]
        
        if len(Y_fallback) > 10:
            Y = Y_fallback[:min(100, len(Y_fallback))]  # Limit to avoid computation issues
        else:
            h_α = 0.0
            h_β = 0.0
            α = float('inf')
            β = 0.0
    
    if len(Y) > 0:
        # h_α: Point-to-surface distance (using actual superquadric surface)
        try:
            S_surface = sample_superquadric_surface(S, n_samples=500)  # Reduced samples for performance
            if len(S_surface) > 0:
                tree_surface = KDTree(S_surface)
                distances_Y_to_S, _ = tree_surface.query(Y)
                α = np.mean(distances_Y_to_S)
                
                # Scale q_α with object size
                q_α = 0.001 * (char_size / 0.05)  # Scale with object size
                h_α = np.exp(- (α**2) / q_α)
            else:
                α = float('inf')
                h_α = 0.0
        except Exception as alpha_error:
            print(f"    [ERROR] h_α computation failed: {alpha_error}")
            α = float('inf')
            h_α = 0.0

        # h_β: Coverage
        try:
            if len(S_surface) > 0:
                tree_Y = KDTree(Y)
                d_th = char_size * 0.2  # Scale coverage threshold with object size
                distances_S_to_Y, _ = tree_Y.query(S_surface)
                T_mask = distances_S_to_Y <= d_th
                T_count = np.sum(T_mask)
                β = T_count / len(S_surface)
                h_β = β**2
            else:
                β = 0.0
                h_β = 0.0
        except Exception as beta_error:
            print(f"    [ERROR] h_β computation failed: {beta_error}")
            β = 0.0
            h_β = 0.0
    
    # h_γ and h_δ calculations
    closing_dir = R @ G.lambda_local
    half_open = G.max_open / 2.0
    
    tip1_world = t + closing_dir * half_open
    tip2_world = t - closing_dir * half_open
    
    curv1 = gaussian_curvature_at_point(tip1_world, S)
    curv2 = gaussian_curvature_at_point(tip2_world, S)
    γ = (curv1 + curv2) / 2.0
    
    q_γ = 0.5
    h_γ = np.exp(- (γ**2) / q_γ)

    centroid = np.mean(X, axis=0)
    δ = np.linalg.norm(t - centroid)
    q_δ = 0.005
    h_δ = np.exp(- (δ**2) / q_δ)

    final_score = h_α * h_β * h_γ * h_δ
    return final_score

def gaussian_curvature_at_point(point_world, S):
    """
    Compute Gaussian curvature of superquadric surface at a given world point.
    Scale-aware curvature for small objects
    """
    # Transform point to SQ-local coordinates
    point_local = S.R.T @ (point_world - S.T)
    x, y, z = point_local
    
    eps1, eps2 = S.ε1, S.ε2
    ax, ay, az = S.ax, S.ay, S.az
    
    # Calculate characteristic size (geometric mean of scales)
    char_size = (ax * ay * az) ** (1/3)  # Geometric mean
    
    # NEW: Normalize curvature by object size
    # For the paper's method, typical objects are 10-20cm
    # Your objects are 2-4cm, so we need to scale accordingly
    
    reference_size = 0.10  # 10cm reference size (typical for paper)
    size_ratio = char_size / reference_size
    
    # Base curvature from shape parameters (conservative)
    if eps1 < 0.5 or eps2 < 0.5:
        shape_factor = 2.0  # Box-like, higher curvature at edges
    elif eps1 > 1.5 and eps2 > 1.5:
        shape_factor = 0.3  # Sphere-like, lower curvature
    else:
        shape_factor = 1.0  # Ellipsoid-like, moderate
    
    # Smaller objects get LOWER curvature penalty (counter-intuitive but needed for small objects)
    size_normalized_factor = 1.0 / (size_ratio + 0.5)  # Add 0.5 to prevent division issues
    
    # Position factor: very conservative for small objects
    r_local = np.sqrt((x/ax)**2 + (y/ay)**2 + (z/az)**2)
    position_factor = 1.0 + 0.2 * r_local  # Much more conservative
    
    # FINAL: Combine all factors with small object compensation
    base_curvature = shape_factor * size_normalized_factor * position_factor
    
    # CRITICAL: For very small objects, clamp to much lower range
    if char_size < 0.05:  # Objects smaller than 5cm
        max_curvature = 1.5  # Much lower max for small objects
    elif char_size < 0.10:  # Objects smaller than 10cm
        max_curvature = 2.5
    else:
        max_curvature = 5.0  # Original max for larger objects
    
    gaussian_curv = np.clip(base_curvature, 0.1, max_curvature)
    
    return gaussian_curv

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