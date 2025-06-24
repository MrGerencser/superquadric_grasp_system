import numpy as np
from typing import List, Tuple, Optional, Any
from scipy.spatial.transform import Rotation as R_simple
from sklearn.cluster import KMeans
import open3d as o3d

# Import your existing modules
from EMS.EMS_recovery import EMS_recovery


def calculate_k_superquadrics(point_count: int, logger=None) -> int:
    """Calculate number of superquadrics K based on point cloud size (from paper equation 12)"""
    X_size = int(point_count)
    
    if X_size < 8000:
        K = 6
        if logger:
            logger.info(f"Point count: {X_size} < 8000, using K = 6")
    else:
        K = 8 + 2 * ((X_size - 8000) // 4000)
        if logger:
            logger.info(f"Point count: {X_size} ≥ 8000, using K = {K}")
    
    # Cap at reasonable maximum
    K_max = 20
    if K > K_max:
        if logger:
            logger.warning(f"Calculated K = {K} exceeds maximum {K_max}, capping to {K_max}")
        K = K_max
    
    return K


def calculate_moment_of_inertia(points: np.ndarray) -> np.ndarray:
    """Calculate moment of inertia tensor for point cloud"""
    center = np.mean(points, axis=0)
    centered_points = points - center
    
    # Calculate inertia tensor
    I = np.zeros((3, 3))
    for p in centered_points:
        I += np.outer(p, p)
    I /= len(points)
    
    return I


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


def validate_superquadric_strict(sq, points: np.ndarray) -> bool:
    """Stricter validation for superquadrics"""
    try:
        # Check scale bounds
        if np.any(sq.scale < 0.003) or np.any(sq.scale > 0.5):
            return False
        
        # Check shape parameters
        if np.any(np.array(sq.shape) < 0.1) or np.any(np.array(sq.shape) > 4.0):
            return False
        
        # Check translation bounds
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


def initialize_multiple_superquadrics(points_for_ems: np.ndarray, K: int = None, logger=None) -> List[dict]:
    """Initialize K+1 superquadrics using K-means clustering"""
    try:
        if K is None:
            K = calculate_k_superquadrics(len(points_for_ems), logger)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(points_for_ems)
        
        initial_superquadrics = []
        
        # Initialize K superquadrics from clusters
        for i in range(K):
            cluster_points = points_for_ems[cluster_labels == i]
            if len(cluster_points) > 10:
                try:
                    cluster_moi = calculate_moment_of_inertia(cluster_points)
                    cluster_ellipsoid = moi_to_ellipsoid_params(cluster_moi / 2.0, logger)
                    
                    initial_superquadrics.append({
                        'translation': np.mean(cluster_points, axis=0),
                        'scale': cluster_ellipsoid['scale'],
                        'shape': [1.0, 1.0],  # Start with ellipsoid
                        'rotation': cluster_ellipsoid['rotation'],
                        'type': f'cluster_{i}'
                    })
                except Exception as cluster_error:
                    if logger:
                        logger.warning(f"Failed to initialize cluster {i}: {cluster_error}")
        
        # Add global superquadric (K+1)
        try:
            global_moi = calculate_moment_of_inertia(points_for_ems)
            global_ellipsoid = moi_to_ellipsoid_params(global_moi / 2.0, logger)
            
            initial_superquadrics.append({
                'translation': np.mean(points_for_ems, axis=0),
                'scale': global_ellipsoid['scale'],
                'shape': [1.0, 1.0],
                'rotation': global_ellipsoid['rotation'],
                'type': 'global'
            })
        except Exception as global_error:
            if logger:
                logger.warning(f"Failed to initialize global superquadric: {global_error}")
        
        return initial_superquadrics
        
    except Exception as e:
        if logger:
            logger.error(f"Error initializing multiple superquadrics: {e}")
        return []


def fit_single_superquadric(points: np.ndarray, outlier_ratio: float = 0.9, logger=None) -> Tuple[Any, List[Any]]:
    """Fit single superquadric using EMS_recovery"""
    try:
        points_center = np.mean(points, axis=0)
        centered_points = points - points_center
        
        sq_candidate, probabilities = EMS_recovery(
            centered_points,
            OutlierRatio=outlier_ratio,
            MaxIterationEM=60,
            ToleranceEM=1e-3,
            RelativeToleranceEM=1e-1,
            MaxOptiIterations=5,
            Sigma=0.02,
            MaxiSwitch=1,
            AdaptiveUpperBound=True,
            Rescale=True
        )
        
        # Translate back to world coordinates
        sq_candidate.translation = sq_candidate.translation + points_center
        
        if validate_superquadric(sq_candidate, points):
            return sq_candidate, [sq_candidate]
        else:
            return None, []
            
    except Exception as e:
        if logger:
            logger.error(f"Error in single superquadric fitting: {e}")
        return None, []


def fit_multiple_superquadrics_ensemble(points_for_ems: np.ndarray, outlier_ratio: float = 0.9, logger=None) -> Tuple[Any, List[Any]]:
    """Implement multiple superquadric fitting"""
    try:
        # Calculate K value based on point count
        K = calculate_k_superquadrics(len(points_for_ems), logger)
        
        # Initialize ellipsoids using K-means clustering
        initial_ellipsoids = initialize_multiple_superquadrics(points_for_ems, K, logger)
        
        if not initial_ellipsoids:
            if logger:
                logger.warning("Failed to initialize ellipsoids")
            return None, []
        
        if logger:
            logger.info(f"Initialized {len(initial_ellipsoids)} valid ellipsoids")
        
        # Fit superquadrics sequentially
        fitted_superquadrics = []
        
        for i, ellipsoid_init in enumerate(initial_ellipsoids):
            try:
                if logger:
                    logger.info(f"Fitting superquadric {i+1}/{len(initial_ellipsoids)}")
                
                # Center points relative to this ellipsoid
                centered_points = points_for_ems - ellipsoid_init['translation']
                
                # Apply rotation if available
                if 'rotation' in ellipsoid_init:
                    centered_points = centered_points @ ellipsoid_init['rotation'].T
                
                # Fit using EMS_recovery
                sq_fitted, probabilities = EMS_recovery(
                    centered_points,
                    OutlierRatio=outlier_ratio,
                    MaxIterationEM=60,
                    ToleranceEM=1e-3,
                    RelativeToleranceEM=1e-1,
                    MaxOptiIterations=5,
                    Sigma=0.02,
                    MaxiSwitch=1,
                    AdaptiveUpperBound=True,
                    Rescale=True,
                    # Initialize with ellipsoid parameters
                    # ellipsoid_scale=ellipsoid_init['scale'],
                    # ellipsoid_shape=ellipsoid_init['shape']
                )
                
                # Transform back to world coordinates
                if 'rotation' in ellipsoid_init:
                    # Apply inverse rotation to translation
                    sq_fitted.translation = ellipsoid_init['rotation'] @ sq_fitted.translation
                sq_fitted.translation = sq_fitted.translation + ellipsoid_init['translation']
                
                # Validate superquadric
                if validate_superquadric_strict(sq_fitted, points_for_ems):
                    coverage = evaluate_sq_coverage(sq_fitted, points_for_ems, logger)
                    if logger:
                        logger.info(f"SQ {i+1}: Valid with coverage {coverage:.3f}")
                    fitted_superquadrics.append(sq_fitted)
                else:
                    if logger:
                        logger.warning(f"SQ {i+1}: Failed validation")
                    
            except Exception as sq_error:
                if logger:
                    logger.warning(f"Failed to fit superquadric {i+1}: {sq_error}")
                continue
        
        if not fitted_superquadrics:
            if logger:
                logger.warning("No valid superquadrics fitted")
            return None, []
        
        # Select best superquadric based on coverage
        best_sq = None
        best_coverage = 0.0
        
        for i, sq in enumerate(fitted_superquadrics):
            coverage = evaluate_sq_coverage(sq, points_for_ems, logger)
            if logger:
                logger.info(f"SQ {i+1}: Coverage = {coverage:.3f}")
            if coverage > best_coverage:
                best_coverage = coverage
                best_sq = sq
        
        if logger:
            logger.info(f"Successfully fitted {len(fitted_superquadrics)} superquadrics, best coverage: {best_coverage:.3f}")
        return best_sq, fitted_superquadrics
        
    except Exception as e:
        if logger:
            logger.error(f"Error in ensemble superquadric fitting: {e}")
            import traceback
            traceback.print_exc()
        return None, []