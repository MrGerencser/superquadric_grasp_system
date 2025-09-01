import numpy as np
from typing import List, Tuple, Optional, Any
from sklearn.cluster import KMeans

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
    RESCALE = True
    t0 = points.mean(axis=0)
    scale = (np.max(points - t0) / 10.0) if RESCALE else 1.0

    # Build the single ellipsoid seed
    seed = ellipsoid_seed(points, tag="global", logger=logger)

    # Convert seed to EMS frame and run EMS
    x0_prior = to_ems_frame(seed, scale=scale, t0=t0)
    sq, _ = EMS_recovery(
        points,
        x0_prior=x0_prior,
        OutlierRatio=outlier_ratio,
        MaxIterationEM=60,
        MaxOptiIterations=5,
        Rescale=RESCALE,
    )

    # Validate & return
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
    
    # Choose K
    K = calculate_k_superquadrics(len(points), logger)

    # Build K + 1 ellipsoid seeds
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points)

    seeds: List[dict] = []

    # K cluster seeds
    for i in range(K):
        cluster_mask = (labels == i)
        cluster_pts = points[cluster_mask]
        if len(cluster_pts) < 50:
            if logger: 
                logger.warning(f"Cluster {i} too small – skipped")
            continue
        seeds.append(ellipsoid_seed(cluster_pts, f"cluster_{i}", logger))

    # Extra global seed
    seeds.append(ellipsoid_seed(points, "global", logger))

    if not seeds:
        if logger: 
            logger.error("No valid seeds – aborting")
        return None, []

    # Convert each seed to EMS coordinates
    t0 = points.mean(axis=0)
    scale = (np.max(points - t0) / 10) if RESCALE else 1.0
    
    # Run EMS once per seed
    results = []
    for seed in seeds:
        x0 = to_ems_frame(seed, scale=scale, t0=t0)
        sq, p = EMS_recovery(
            points,
            x0_prior=x0,
            OutlierRatio=outlier_ratio,
            MaxIterationEM=60,
            MaxOptiIterations=5,
            Rescale=RESCALE,
        )
        results.append((sq, p, seed["type"]))

    if not results:
        return None, []

    # Pick the best SQ by coverage
    best_tuple = max(results, key=lambda r: evaluate_sq_coverage(r[0], points))
    best_sq = best_tuple[0]
    all_sqs = [r[0] for r in results]

    return best_sq, all_sqs

# ---------------------------------------UTILITIES-------------------------------------------
def calculate_k_superquadrics(n_pts: int, logger=None) -> int:
    """Equation 12 in the paper, capped to MAX_ALLOWED_K."""
    if n_pts < 8000:
        k = max(1, int(np.log10(n_pts)) - 2)
    else:
        k = 2
    k = min(k, 20)  # cap to max 20 superquadrics
    if logger:
        logger.debug(f"Calculated K={k} for {n_pts} points")
    return k
    
def ellipsoid_seed(pts: np.ndarray, tag: str, logger=None) -> dict:
    moi = calculate_moment_of_inertia(pts) / 2.0  # MoI halved
    ell = moi_to_ellipsoid_params(moi, logger)
    seed = {
        "translation": pts.mean(axis=0),
        "semi_axes": ell["scale"],
        "shape": [1.0, 1.0],
        "rotation": ell["rotation"],
        "type": tag
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
        
        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Convert eigenvalues to ellipsoid semi-axes
        # For ellipsoid: I = (2/5) * m * diag(b²+c², a²+c², a²+b²)
        # Solving: a² = (I_yy + I_zz - I_xx) / (2/5 * m)
        scale = np.sqrt(np.maximum(eigenvals, 1e-8))
        
        return {
            "scale": scale,
            "rotation": eigenvecs
        }
        
    except Exception as e:
        if logger:
            logger.warning(f"MoI conversion failed: {e}, using default")
        return {
            "scale": np.array([0.05, 0.05, 0.05]),
            "rotation": np.eye(3)
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
        # Simple distance-based coverage metric
        # Transform points to SQ local frame and compute fit
        points_local = sq.world_to_obj(points) if hasattr(sq, 'world_to_obj') else points
        
        # Compute distances or some coverage metric
        # This is a placeholder - implement based on your SQ distance function
        distances = np.linalg.norm(points_local, axis=1)
        coverage = np.mean(np.exp(-distances))
        
        return coverage
        
    except Exception as e:
        if logger:
            logger.warning(f"Coverage evaluation failed: {e}")
        return 0.0