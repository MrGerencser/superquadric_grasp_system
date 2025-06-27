import numpy as np
from typing import List, Tuple, Optional, Any
from sklearn.cluster import KMeans
import open3d as o3d

# Import your existing modules
from EMS.EMS_recovery import EMS_recovery, RotM2Euler

def fit_single_superquadric(
    points: np.ndarray,
    outlier_ratio: float = 0.9,
    logger=None
) -> Tuple[Any, List[Any]]:
    """
    Fit **one** superquadric to `points` using exactly the same
    notation and helper functions that the ensemble version uses.
    Returns (best_sq, [best_sq]) to keep the API uniform.
    """
    RESCALE   = True
    t0        = points.mean(axis=0)
    scale     = (np.max(points - t0) / 10.0) if RESCALE else 1.0

    # ------------------------------------------------------------------
    # 1.  Build the single ellipsoid seed (MoI/2 rule, like the paper)
    # ------------------------------------------------------------------
    seed = ellipsoid_seed(points, tag="global", logger=logger)

    # ------------------------------------------------------------------
    # 2.  Convert seed to EMS frame and run EMS
    # ------------------------------------------------------------------
    x0_prior = to_ems_frame(seed, scale=scale, t0=t0)
    sq, _ = EMS_recovery(
        points,
        x0_prior         = x0_prior,
        OutlierRatio     = outlier_ratio,
        MaxIterationEM   = 60,
        MaxOptiIterations= 5,
        Rescale          = RESCALE,
    )

    # ------------------------------------------------------------------
    # 3.  Validate & return in the same shape as the ensemble routine
    # ------------------------------------------------------------------
    if validate_superquadric(sq, points):
        return sq, [sq]

    if logger:
        logger.warning("Single-SQ fit did not pass validation")
    return None, []

def fit_multiple_superquadrics(
    points: np.ndarray,
    outlier_ratio: float = 0.9,
    logger=None
) -> Tuple[Any, List[Any]]:
    """
    Recover K + 1 superquadrics (K clusters + 1 global) and
    return the best-coverage SQ plus the full list.
    """
    RESCALE = True
    # ------------------------------------------------------------------
    # 1.  Choose K (your own heuristic)
    # ------------------------------------------------------------------
    K = calculate_k_superquadrics(len(points), logger)

    # ------------------------------------------------------------------
    # 2.  Build K + 1 ellipsoid seeds exactly as in the paper
    # ------------------------------------------------------------------
    kmeans  = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels  = kmeans.fit_predict(points)

    seeds: List[dict] = []

    # K cluster seeds
    for i in range(K):
        cluster_pts = points[labels == i]
        if len(cluster_pts) < 10:
            if logger: logger.warning(f"Cluster {i} too small – skipped")
            continue
        seeds.append(ellipsoid_seed(cluster_pts, f"cluster_{i}", logger))

    # extra global seed
    seeds.append(ellipsoid_seed(points, "global", logger))

    if not seeds:
        if logger: logger.error("No valid seeds – aborting")
        return None, []

    # ------------------------------------------------------------------
    # 3.  Convert each seed to EMS coordinates (centred & scaled)
    # ------------------------------------------------------------------
    t0    = points.mean(axis=0)
    scale = (np.max(points - t0) / 10) if RESCALE else 1.0
    # ------------------------------------------------------------------
    # 4.  Run EMS once per seed
    # ------------------------------------------------------------------
    results = []
    for seed in seeds:
        x0 = to_ems_frame(seed, scale=scale, t0=t0)
        sq, p = EMS_recovery(
            points,
            x0_prior         = x0,
            OutlierRatio     = outlier_ratio,   # <- use caller’s value
            MaxIterationEM   = 60,
            MaxOptiIterations= 5,
            Rescale          = RESCALE,
        )
        results.append((sq, p, seed["type"]))

    if not results:
        return None, []

    # ------------------------------------------------------------------
    # 5.  Pick the best SQ by coverage
    # ------------------------------------------------------------------
    best_tuple = max(results, key=lambda r: evaluate_sq_coverage(r[0], points))
    best_sq    = best_tuple[0]      # keep only the superquadric object
    all_sqs    = [r[0] for r in results]

    return best_sq, all_sqs


# ---------------------------------------UTILITIES-------------------------------------------
def calculate_k_superquadrics(n_pts: int, logger=None) -> int:
    """Equation 12 in the paper, capped to MAX_ALLOWED_K."""
    if n_pts < 8000:
        k = 6
    else:
        k = 8 + 2 * ((n_pts - 8000) // 4000)
    k = min(k, 20)  # cap to max 20 superquadrics
    if logger:
        logger.info(f"Point count {n_pts}, choosing K = {k}")
    return k
    
def ellipsoid_seed(pts: np.ndarray, tag: str, logger=None) -> dict:
    moi   = calculate_moment_of_inertia(pts) / 2.0      # ✱ MoI halved
    ell   = moi_to_ellipsoid_params(moi, logger)        # → scale & rotation
    seed  = {
        "translation": pts.mean(axis=0),                # world coords
        "semi_axes":   ell["scale"],                    # a1,a2,a3  (world units)
        "shape":       [1.0, 1.0],                      # ellipsoid
        "rotation":    ell["rotation"],                 # 3×3 matrix
        "type":        tag
    }
    return seed

def to_ems_frame(seed: dict, scale: float, t0: np.ndarray) -> np.ndarray:
    a1, a2, a3 = np.asarray(seed["semi_axes"]) / scale
    tx, ty, tz = (seed["translation"] - t0) / scale
    e1, e2, e3 = RotM2Euler(seed["rotation"])
    return np.array([1.0, 1.0, a1, a2, a3, e1, e2, e3, tx, ty, tz])

def moi_to_ellipsoid_params(inertia_tensor: np.ndarray, logger=None) -> dict:
    """Convert moment of inertia to ellipsoid parameters"""
    try:
        eigenvals, eigenvecs = np.linalg.eigh(inertia_tensor)
        
        # Ensure eigenvalues are positive
        eigenvals = np.abs(eigenvals)
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        # Scale from eigenvalues
        scale = np.sqrt(eigenvals)
        
        # Ensure proper rotation matrix
        rotation_matrix = eigenvecs.copy()
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, -1] *= -1
        
        return {
            'scale': scale,
            'rotation': rotation_matrix
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error in moi_to_ellipsoid_params: {e}")
        return {
            'scale': np.array([0.01, 0.01, 0.01]),
            'rotation': np.eye(3)
        }
        
        
def calculate_moment_of_inertia(x: np.ndarray) -> np.ndarray:
    """Vectorised 3×3 MoI tensor of point set `x`."""
    centred = x - x.mean(0)
    return centred.T @ centred / x.shape[0]

def validate_superquadric(sq, points: np.ndarray) -> bool:
    """Validate superquadric parameters"""
    try:
        # Check scale bounds
        if np.any(sq.scale < 0.003) or np.any(sq.scale > 0.5):
            return False
        
        # Check shape parameters
        if np.any(np.array(sq.shape) < 0.1) or np.any(np.array(sq.shape) > 4.0):
            return False
        
        # Check translation is reasonable
        point_center = np.mean(points, axis=0)
        point_bounds = np.max(points, axis=0) - np.min(points, axis=0)
        max_deviation = np.linalg.norm(point_bounds) * 0.5
        
        if np.linalg.norm(sq.translation - point_center) > max_deviation:
            return False
        
        return True
        
    except Exception:
        return False

def evaluate_sq_coverage(sq, points: np.ndarray, logger=None) -> float:
    """Evaluate how well a superquadric covers the point cloud"""
    try:
        # Transform points to superquadric coordinate system
        centered_points = points - sq.translation
        
        # Rotate points if rotation matrix available
        if hasattr(sq, 'RotM'):
            rotated_points = centered_points @ sq.RotM.T
        else:
            rotated_points = centered_points
        
        # Calculate superquadric function
        safe_scale = np.maximum(sq.scale, 1e-6)
        normalized = np.abs(rotated_points) / safe_scale
        
        eps1, eps2 = sq.shape
        eps1 = max(eps1, 0.1)
        eps2 = max(eps2, 0.1)
        
        # Superquadric function evaluation
        xy_term = (normalized[:, 0]**(2/eps2) + normalized[:, 1]**(2/eps2))**(eps2/eps1)
        z_term = normalized[:, 2]**(2/eps1)
        F_values = xy_term + z_term
        
        # Coverage calculation
        inside_mask = F_values <= 1.2
        coverage = np.sum(inside_mask) / len(points)
        
        return coverage
        
    except Exception as e:
        if logger:
            logger.warning(f"Error evaluating SQ coverage: {e}")
        return 0.0