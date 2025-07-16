# superquadric_grasp_system/visualization/main_visualizer.py 
import os, sys
# so that "superquadric_grasp_system" is on PYTHONPATH when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Any, Tuple, Union
from scipy.spatial.transform import Rotation as R_simple
import traceback

# Import from your existing modules
# from ..utils.grasp_planning.geometric_primitives import Gripper, Superquadric
from superquadric_grasp_system.utils.superquadric_grasp_planning.geometric_primitives import Gripper, Superquadric

# =============================================================================
# GRASP VISUALIZATION
# =============================================================================

class PerceptionVisualizer:
    """Unified perception visualization handler"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.default_gripper = Gripper()
        
        # Visualization settings
        self.gripper_colors = [
            (0.2, 0.8, 0.2), (0.8, 0.2, 0.2), (0.2, 0.2, 0.8), 
            (0.8, 0.8, 0.2), (0.8, 0.2, 0.8), (0.2, 0.8, 0.8)
        ]
        
        self.superquadric_colors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], 
            [1, 1, 0], [1, 0, 1], [0, 1, 1]
        ]
        
        # Filter visualization colors
        self.filter_colors = {
            'collision': (1.0, 0.0, 0.0),      # Red
            'support': (0.0, 1.0, 0.0),        # Green
            'quality': (0.0, 0.0, 1.0),        # Blue
            'final': (1.0, 0.5, 0.0),          # Orange
            'rejected': (0.5, 0.5, 0.5)        # Gray
        }
        
    # =============================================================================
    # DETECTED CLOUD VISUALIZATION
    # =============================================================================
    
    def visualize_detected_cloud_filtering(self, original_points: np.ndarray, filtered_points: np.ndarray, 
                                         class_id: int, object_center: np.ndarray = None,
                                         window_name: str = None) -> None:
        """Visualize before/after point cloud filtering for detected objects"""
        try:
            if object_center is None:
                object_center = np.mean(original_points, axis=0) if len(original_points) > 0 else np.array([0, 0, 0])
            
            # Create original point cloud (red)
            original_pcd = o3d.geometry.PointCloud()
            original_pcd.points = o3d.utility.Vector3dVector(original_points)
            original_pcd.paint_uniform_color([1, 0, 0])  # Red for original
            
            # Create filtered point cloud (green)
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            filtered_pcd.paint_uniform_color([0, 1, 0])  # Green for filtered
            
            # Add coordinate frame at object center
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            coord_frame.translate(object_center)
            
            # Set window name
            if window_name is None:
                window_name = f"Before/After Filtering - Class {class_id} (Original: {len(original_points)}, Filtered: {len(filtered_points)})"
            
            # Visualize both point clouds together
            o3d.visualization.draw_geometries(
                [original_pcd, filtered_pcd, coord_frame],
                window_name=window_name,
                zoom=0.7,
                front=[0, -1, 0],
                lookat=object_center,
                up=[0, 0, 1]
            )
            
        except Exception as e:
            print(f"Error in detected cloud visualization: {e}")
    
    # =============================================================================
    # CORE GRASP VISUALIZATION
    # =============================================================================
    
    def get_gripper_meshes(self, grasp_transform: np.ndarray, gripper: Gripper = None, 
                          show_sweep_volume: bool = False, color: Tuple[float, float, float] = (0.2, 0.8, 0), 
                          finger_tip_to_origin: bool = True) -> List[o3d.geometry.TriangleMesh]:
        """Create gripper visualization meshes"""
        if gripper is None:
            gripper = self.default_gripper
        
        # Get gripper meshes from geometric primitives
        gripper_meshes = gripper.make_open3d_meshes(colour=color)
        
        # Handle different mesh configurations
        if len(gripper_meshes) == 4:
            finger_L, finger_R, connector, back_Z = gripper_meshes
            all_gripper_parts = [finger_L, finger_R, back_Z, connector]
        else:
            all_gripper_parts = gripper_meshes
        
        # Transform all meshes to world coordinates
        meshes = []
        for mesh in all_gripper_parts:
            mesh_world = mesh.transform(grasp_transform.copy())
            meshes.append(mesh_world)
        
        # Add coordinate frame at grasp pose
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        coord_frame.transform(grasp_transform)
        meshes.append(coord_frame)
        
        # Add directional arrows
        meshes.extend(self._create_direction_arrows(grasp_transform, gripper))
        
        # Add sweep volume if requested
        if show_sweep_volume:
            sweep_volume = self._create_sweep_volume(grasp_transform, gripper)
            if sweep_volume:
                meshes.append(sweep_volume)
        
        return meshes
    
    def _create_direction_arrows(self, grasp_transform: np.ndarray, gripper: Gripper) -> List[o3d.geometry.TriangleMesh]:
        """Create directional arrows for grasp visualization"""
        arrows = []
        
        # Approach direction arrow (blue)
        arrow_length = 0.04
        approach_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.002, cone_radius=0.004,
            cylinder_height=arrow_length * 0.7, cone_height=arrow_length * 0.3
        )
        
        # Arrow points along gripper's approach direction (-Z in gripper frame)
        approach_dir = grasp_transform[:3, :3] @ gripper.approach_axis
        arrow_pos = grasp_transform[:3, 3] + approach_dir * arrow_length
        
        # Orient arrow along approach direction
        z_axis = np.array([0, 0, 1])
        if not np.allclose(approach_dir, z_axis):
            if np.allclose(approach_dir, -z_axis):
                arrow_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                v = np.cross(z_axis, approach_dir)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, approach_dir)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                arrow_rotation = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
        else:
            arrow_rotation = np.eye(3)
        
        arrow_transform = np.eye(4)
        arrow_transform[:3, :3] = arrow_rotation
        arrow_transform[:3, 3] = arrow_pos
        approach_arrow.transform(arrow_transform)
        approach_arrow.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
        arrows.append(approach_arrow)
        
        # Closing direction arrow (red)
        closing_dir = grasp_transform[:3, :3] @ gripper.lambda_local  # Y axis
        closing_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.002, cone_radius=0.004,
            cylinder_height=0.03, cone_height=0.008
        )
        
        # Orient closing arrow
        if not np.allclose(closing_dir, z_axis):
            if np.allclose(closing_dir, -z_axis):
                closing_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                v = np.cross(z_axis, closing_dir)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, closing_dir)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                closing_rotation = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
        else:
            closing_rotation = np.eye(3)
        
        closing_arrow_transform = np.eye(4)
        closing_arrow_transform[:3, :3] = closing_rotation
        closing_arrow_transform[:3, 3] = grasp_transform[:3, 3] + closing_dir * 0.03
        closing_arrow.transform(closing_arrow_transform)
        closing_arrow.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        arrows.append(closing_arrow)
        
        return arrows
    
    def _create_sweep_volume(self, grasp_transform: np.ndarray, gripper: Gripper) -> Optional[o3d.geometry.TriangleMesh]:
        """Create sweep volume visualization"""
        try:
            sweep_volume = o3d.geometry.TriangleMesh.create_box(
                width=gripper.thickness * 2,
                height=gripper.max_open,
                depth=gripper.jaw_len
            )
            
            # Center the sweep volume in gripper local coordinates
            sweep_volume.translate([
                -gripper.thickness,
                -gripper.max_open / 2,
                -gripper.jaw_len
            ])
            
            # Apply the grasp transformation
            sweep_transform = grasp_transform.copy()
            sweep_volume.transform(sweep_transform)
            
            # Make it semi-transparent blue
            sweep_volume.paint_uniform_color([0.2, 0.2, 0.8])
            
            return sweep_volume
            
        except Exception as e:
            print(f"Error creating sweep volume: {e}")
            return None
    
    # =============================================================================
    # SUPERQUADRIC FIT VISUALIZATION
    # =============================================================================
    
    def visualize_superquadric_fit(self, points: np.ndarray, all_sqs: List[Any], 
                                class_id: int, class_names: Dict[int, str] = None,
                                window_name: str = None) -> int:
        """
        Enhanced superquadric fitting visualization
        
        Args:
            points: Original point cloud points
            all_sqs: List of fitted superquadric objects
            class_id: Object class ID  
            class_names: Mapping of class IDs to names
            window_name: Custom window name
            
        Returns:
            Number of successfully visualized superquadrics
        """
        try:
            import colorsys
            
            # Create point cloud from original points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for original points
            
            # Create list of geometries to visualize
            geometries = [pcd]
            
            # Generate distinct colors for each superquadric (using golden angle)
            colors = []
            for i in range(len(all_sqs)):
                hue = (i * 137.5) % 360  # Golden angle for good distribution
                saturation = 0.8
                value = 0.9
                # Convert HSV to RGB
                rgb = colorsys.hsv_to_rgb(hue/360.0, saturation, value)
                colors.append(list(rgb))
            
            # Add each superquadric mesh with different colors
            successful_quadrics = 0
            for i, sq in enumerate(all_sqs):
                try:
                    # Create superquadric mesh using the centralized method
                    mesh = self._superquadric_to_open3d_mesh(sq)
                    if mesh is not None:
                        color = colors[i % len(colors)]
                        mesh.paint_uniform_color(color)
                        mesh.compute_vertex_normals()
                        geometries.append(mesh)
                        successful_quadrics += 1
                        
                        # Log quadric parameters (like in superquadric_grasp_node.py)
                        print(f"Quadric {i+1}: shape=({sq.shape[0]:.3f}, {sq.shape[1]:.3f}), "
                            f"scale=({sq.scale[0]:.3f}, {sq.scale[1]:.3f}, {sq.scale[2]:.3f})")
                        
                except Exception as mesh_error:
                    print(f"Failed to create mesh for quadric {i+1}: {mesh_error}")
            
            print(f"Adaptive multiquadric fitting: {successful_quadrics}/{len(all_sqs)} successful meshes")
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            geometries.append(coord_frame)
            
            # Get object class name
            if class_names is None:
                class_names = {0: 'Cone', 1: 'Cup', 2: 'Mallet', 3: 'Screw Driver', 4: 'Sunscreen'}
            class_name = class_names.get(class_id, f'Class_{class_id}')
            
            # Set window name (matching superquadric_grasp_node.py style)
            if window_name is None:
                window_name = f"Adaptive Multi-Superquadric Fitting (K={len(all_sqs)})"
            
            # Visualize all geometries together
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_name,
                zoom=0.7,
                front=[0, -1, 0],
                lookat=pcd.get_center() if len(pcd.points) > 0 else [0, 0, 0],
                up=[0, 0, 1]
            )
            
            return successful_quadrics
            
        except Exception as e:
            print(f"Error in superquadric fit visualization: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _superquadric_to_open3d_mesh(self, sq):
        """Convert superquadric object to Open3D mesh"""
        try:
            # Try to use EMS utilities if available
            from EMS.utilities import uniformSampledSuperellipse, create_mesh_from_grid
            
            # Avoid numerical instability in sampling
            shape_stable = sq.shape.copy()
            if shape_stable[0] < 0.007:
                shape_stable[0] = 0.007
            if shape_stable[1] < 0.007:
                shape_stable[1] = 0.007
            
            # Sampling points in superellipse    
            point_eta = uniformSampledSuperellipse(shape_stable[0], [1, sq.scale[2]], 1e-2, 10000, 0.2)
            point_omega = uniformSampledSuperellipse(shape_stable[1], [sq.scale[0], sq.scale[1]], 1e-2, 10000, 0.2)
            
            # Preallocate meshgrid
            x_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
            y_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
            z_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))

            for m in range(np.shape(point_omega)[1]):
                for n in range(np.shape(point_eta)[1]):
                    point_temp = np.zeros(3)
                    point_temp[0:2] = point_omega[:, m] * point_eta[0, n]
                    point_temp[2] = point_eta[1, n]
                    point_temp = sq.RotM @ point_temp + sq.translation

                    x_mesh[m, n] = point_temp[0]
                    y_mesh[m, n] = point_temp[1]
                    z_mesh[m, n] = point_temp[2]
            
            # Create Open3D mesh from the grid
            mesh = create_mesh_from_grid(x_mesh, y_mesh, z_mesh)
            return mesh
            
        except ImportError:
            print("EMS utilities not available - using fallback mesh creation")
            return self._create_superquadric_approximation_from_object(sq)
        except Exception as e:
            print(f"Error creating superquadric mesh: {e}")
            return None

    def _create_superquadric_approximation_from_object(self, sq):
        """Create an approximate superquadric mesh using ellipsoid from superquadric object"""
        try:
            # Create ellipsoid as approximation
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
            
            # Scale to match superquadric dimensions
            mesh.scale(sq.scale[0], center=(0, 0, 0))
            mesh.scale([1, sq.scale[1]/sq.scale[0], sq.scale[2]/sq.scale[0]], center=(0, 0, 0))
            
            # Rotate using the superquadric's rotation matrix
            mesh.rotate(sq.RotM, center=(0, 0, 0))
            
            # Translate
            mesh.translate(sq.translation)
            
            return mesh
            
        except Exception as e:
            print(f"Error creating superquadric approximation: {e}")
            return None

    # =============================================================================
    # SUPERQUADRIC VISUALIZATION
    # =============================================================================
    
    def create_superquadric_mesh(self, shape: np.ndarray, scale: np.ndarray, 
                               euler: np.ndarray, translation: np.ndarray) -> Optional[o3d.geometry.TriangleMesh]:
        """Create superquadric mesh for visualization"""
        try:
            # Try to use EMS utilities if available
            from EMS.utilities import uniformSampledSuperellipse, create_mesh_from_grid
            
            # Avoid numerical instability in sampling
            shape_stable = shape.copy()
            if shape_stable[0] < 0.007:
                shape_stable[0] = 0.007
            if shape_stable[1] < 0.007:
                shape_stable[1] = 0.007
            
            # Sampling points in superellipse    
            point_eta = uniformSampledSuperellipse(shape_stable[0], [1, scale[2]], 1e-2, 10000, 0.2)
            point_omega = uniformSampledSuperellipse(shape_stable[1], [scale[0], scale[1]], 1e-2, 10000, 0.2)
            
            # Create rotation matrix from Euler angles
            R = R_simple.from_euler('xyz', euler).as_matrix()
            
            # Preallocate meshgrid
            x_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
            y_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
            z_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))

            for m in range(np.shape(point_omega)[1]):
                for n in range(np.shape(point_eta)[1]):
                    point_temp = np.zeros(3)
                    point_temp[0:2] = point_omega[:, m] * point_eta[0, n]
                    point_temp[2] = point_eta[1, n]
                    point_temp = R @ point_temp + translation

                    x_mesh[m, n] = point_temp[0]
                    y_mesh[m, n] = point_temp[1]
                    z_mesh[m, n] = point_temp[2]
            
            # Create Open3D mesh from the grid
            mesh = create_mesh_from_grid(x_mesh, y_mesh, z_mesh)
            return mesh
            
        except ImportError:
            print("Warning: EMS utilities not available. Creating approximation.")
            return self._create_superquadric_approximation(shape, scale, euler, translation)
        except Exception as e:
            print(f"Error creating superquadric mesh: {e}")
            return None
    
    def _create_superquadric_approximation(self, shape: np.ndarray, scale: np.ndarray, 
                                         euler: np.ndarray, translation: np.ndarray) -> Optional[o3d.geometry.TriangleMesh]:
        """Create an approximate superquadric mesh using basic primitives"""
        try:
            # Create ellipsoid as approximation
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
            
            # Scale to match superquadric dimensions
            mesh.scale(scale[0], center=(0, 0, 0))
            mesh.scale([1, scale[1]/scale[0], scale[2]/scale[0]], center=(0, 0, 0))
            
            # Rotate
            R = R_simple.from_euler('xyz', euler).as_matrix()
            mesh.rotate(R, center=(0, 0, 0))
            
            # Translate
            mesh.translate(translation)
            
            return mesh
        except Exception as e:
            print(f"Error creating superquadric approximation: {e}")
            return None

    # =============================================================================
    # FILTER VISUALIZATION
    # =============================================================================
    
    def visualize_support_test(self, point_cloud: np.ndarray, grasp_pose: np.ndarray, 
                                       support_test_data: Dict, window_name: str = "Support Test") -> None:
        """Support test visualization with cylinders, support points, etc."""
        try:
            geometries = []
            
            # Extract data
            R_world = support_test_data['R_world']
            t_world = support_test_data['t_world']
            G = support_test_data['G']
            closing_dir = support_test_data['closing_dir']
            tip1 = support_test_data['tip1']
            tip2 = support_test_data['tip2']
            r_support = support_test_data['r_support']
            h_support = support_test_data['h_support']
            mask1 = support_test_data['mask1']
            mask2 = support_test_data['mask2']
            cnt1 = support_test_data['cnt1']
            cnt2 = support_test_data['cnt2']
            required_points = support_test_data['required_points']
            support_result = support_test_data['support_result']
            
            X = point_cloud
            
            # 1. Background points (not supporting either tip)
            background_mask = ~(mask1 | mask2)
            if np.any(background_mask):
                pcd_background = o3d.geometry.PointCloud()
                pcd_background.points = o3d.utility.Vector3dVector(X[background_mask])
                pcd_background.paint_uniform_color([0.7, 0.7, 0.7])  # Gray background
                geometries.append(pcd_background)

            # 2. Support points for tip1
            if np.any(mask1):
                tip1_support_points = X[mask1]
                pcd_tip1 = o3d.geometry.PointCloud()
                pcd_tip1.points = o3d.utility.Vector3dVector(tip1_support_points)
                # Color based on whether tip1 has enough support
                if cnt1 >= required_points:
                    pcd_tip1.paint_uniform_color([0.0, 1.0, 0.0])  # Green = sufficient support
                else:
                    pcd_tip1.paint_uniform_color([1.0, 0.5, 0.0])  # Orange = insufficient support
                geometries.append(pcd_tip1)

            # 3. Support points for tip2
            if np.any(mask2):
                tip2_support_points = X[mask2]
                pcd_tip2 = o3d.geometry.PointCloud()
                pcd_tip2.points = o3d.utility.Vector3dVector(tip2_support_points)
                # Color based on whether tip2 has enough support
                if cnt2 >= required_points:
                    pcd_tip2.paint_uniform_color([0.0, 0.8, 0.0])  # Slightly different green
                else:
                    pcd_tip2.paint_uniform_color([1.0, 0.3, 0.0])  # Slightly different orange
                geometries.append(pcd_tip2)

            # 4. Support points for BOTH tips (overlap)
            overlap_mask = mask1 & mask2
            if np.any(overlap_mask):
                overlap_points = X[overlap_mask]
                pcd_overlap = o3d.geometry.PointCloud()
                pcd_overlap.points = o3d.utility.Vector3dVector(overlap_points)
                pcd_overlap.paint_uniform_color([0.0, 0.0, 1.0])  # Blue = supporting both tips
                geometries.append(pcd_overlap)

            # 5. Support cylinder visualization for tip1
            cylinder1 = o3d.geometry.TriangleMesh.create_cylinder(
                radius=r_support, 
                height=2 * h_support,
                resolution=20
            )
            
            # Orient and position cylinder1
            z_axis = np.array([0, 0, 1])
            if not np.allclose(closing_dir, z_axis):
                if np.allclose(closing_dir, -z_axis):
                    cyl_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
                else:
                    v = np.cross(z_axis, closing_dir)
                    s = np.linalg.norm(v)
                    c = np.dot(z_axis, closing_dir)
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    cyl_rotation = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
            else:
                cyl_rotation = np.eye(3)

            cylinder1_transform = np.eye(4)
            cylinder1_transform[:3, :3] = cyl_rotation
            cylinder1_transform[:3, 3] = tip1
            cylinder1.transform(cylinder1_transform)

            # Create wireframe version for tip1
            cylinder1_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder1)
            if cnt1 >= required_points:
                cylinder1_wireframe.paint_uniform_color([0.0, 0.8, 0.0])  # Green wireframe
            else:
                cylinder1_wireframe.paint_uniform_color([1.0, 0.0, 0.0])  # Red wireframe
            geometries.append(cylinder1_wireframe)

            # 6. Support cylinder visualization for tip2
            cylinder2 = o3d.geometry.TriangleMesh.create_cylinder(
                radius=r_support, 
                height=2 * h_support,
                resolution=20
            )
            
            cylinder2_transform = np.eye(4)
            cylinder2_transform[:3, :3] = cyl_rotation
            cylinder2_transform[:3, 3] = tip2
            cylinder2.transform(cylinder2_transform)

            # Create wireframe version for tip2
            cylinder2_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder2)
            if cnt2 >= required_points:
                cylinder2_wireframe.paint_uniform_color([0.0, 0.6, 0.0])  # Slightly different green
            else:
                cylinder2_wireframe.paint_uniform_color([0.8, 0.0, 0.0])  # Slightly different red
            geometries.append(cylinder2_wireframe)

            # 7. Grasp center as sphere
            grasp_center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
            grasp_center_sphere.translate(t_world)
            grasp_center_sphere.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta
            geometries.append(grasp_center_sphere)

            # 8. Closing direction arrow
            arrow_length = h_support * 2.5
            arrow_end = t_world + closing_dir * arrow_length
            
            # Create line for arrow shaft
            line_points = [t_world, arrow_end]
            line_lines = [[0, 1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(line_lines)
            line_set.paint_uniform_color([1.0, 1.0, 1.0])  # White
            geometries.append(line_set)

            # 9. Gripper geometry
            try:
                gripper_meshes = self.get_gripper_meshes(grasp_pose, gripper=G, color=(0.2, 0.8, 0.2))
                geometries.extend(gripper_meshes)
            except Exception as gripper_error:
                print(f"    [ERROR] Could not add gripper visualization: {gripper_error}")
            
            # 10. Main coordinate frame
            main_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            geometries.append(main_coord_frame)
            
            # Print legend
            print(f"SUPPORT TEST VISUALIZATION:")
            print(f"  🟫 Gray:    Background points (not supporting)")
            tip1_status = "✅ PASS" if cnt1 >= required_points else "❌ FAIL"
            tip2_status = "✅ PASS" if cnt2 >= required_points else "❌ FAIL"
            print(f"  🟢 Green:   Tip1 support points ({cnt1}/{required_points}) {tip1_status}")
            print(f"  🟢 Lt.Green: Tip2 support points ({cnt2}/{required_points}) {tip2_status}")
            print(f"  🟣 Magenta: Grasp center")
            print(f"  ⚪ White:   Closing direction")
            print(f"  🤖 Robot:   Gripper geometry")
            print(f"  🟢 Green cylinders: Sufficient support")
            print(f"  🔴 Red cylinders:   Insufficient support")
            print(f"{'-'*60}")
            print(f"  Support cylinder radius: {r_support:.3f}m")
            print(f"  Support cylinder height: {2*h_support:.3f}m")
            print(f"  Required points per tip: {required_points}")
            overall_result = "✅ PASS" if support_result else "❌ FAIL"
            print(f"  OVERALL SUPPORT TEST: {overall_result}")
            
            # Show visualization
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_name,
                zoom=0.6,
                front=[0, -1, 0],
                lookat=t_world,
                up=[0, 0, 1]
            )
            
        except Exception as e:
            print(f"Error in support test visualization: {e}")
            import traceback
            traceback.print_exc()

    def visualize_collision_test(self, point_cloud: np.ndarray, grasp_pose: np.ndarray,
                                         collision_test_data: Dict, window_name: str = "Collision Test") -> None:
        """Collision test visualization with slabs, collision points, etc."""
        try:
            geometries = []
            
            # Extract data
            R_world = collision_test_data['R_world']
            t_world = collision_test_data['t_world']
            G = collision_test_data['G']
            lambda_dir = collision_test_data['lambda_dir']
            radius = collision_test_data['radius']
            half_height_cylinder = collision_test_data['half_height_cylinder']
            half_height_finger = collision_test_data['half_height_finger']
            small_slab_mask = collision_test_data['small_slab_mask']
            large_slab_mask = collision_test_data['large_slab_mask']
            outside_cylinder_mask_small = collision_test_data['outside_cylinder_mask_small']
            palm_collisions = collision_test_data['palm_collisions']
            finger_collisions = collision_test_data['finger_collisions']
            all_potential_collision_points = collision_test_data['all_potential_collision_points']
            collision_result = collision_test_data['collision_result']
            
            X = point_cloud
            
            # 1. Background points (not in any slab)
            background_mask = ~large_slab_mask
            if np.any(background_mask):
                pcd_background = o3d.geometry.PointCloud()
                pcd_background.points = o3d.utility.Vector3dVector(X[background_mask])
                pcd_background.paint_uniform_color([0.7, 0.7, 0.7])  # Gray background
                geometries.append(pcd_background)

            # 2. Small slab points (cylinder region)
            if np.any(small_slab_mask):
                small_slab_points = X[small_slab_mask]
                
                # Points INSIDE cylinder = GREEN (safe for grasping)
                inside_cylinder_points = small_slab_points[~outside_cylinder_mask_small]
                if len(inside_cylinder_points) > 0:
                    pcd_safe = o3d.geometry.PointCloud()
                    pcd_safe.points = o3d.utility.Vector3dVector(inside_cylinder_points)
                    pcd_safe.paint_uniform_color([0.0, 1.0, 0.0])  # Green = safe
                    geometries.append(pcd_safe)
                
                # Points OUTSIDE cylinder in small slab = POTENTIAL COLLISION (orange)
                if np.any(outside_cylinder_mask_small):
                    potential_collision_points = small_slab_points[outside_cylinder_mask_small]
                    pcd_potential = o3d.geometry.PointCloud()
                    pcd_potential.points = o3d.utility.Vector3dVector(potential_collision_points)
                    pcd_potential.paint_uniform_color([1.0, 0.5, 0.0])  # Orange = potential collision
                    geometries.append(pcd_potential)

            # 3. Finger region points (large slab minus small slab)
            finger_region_mask = collision_test_data['finger_region_mask']
            if np.any(finger_region_mask):
                finger_region_points = X[finger_region_mask]
                pcd_finger_region = o3d.geometry.PointCloud()
                pcd_finger_region.points = o3d.utility.Vector3dVector(finger_region_points)
                pcd_finger_region.paint_uniform_color([0.0, 0.0, 1.0])  # Blue = finger region
                geometries.append(pcd_finger_region)

            # 4. Collision points visualization
            if len(all_potential_collision_points) > 0:
                # PALM COLLISION POINTS (red spheres)
                if len(palm_collisions) > 0 and np.any(palm_collisions):
                    palm_collision_points = all_potential_collision_points[palm_collisions]
                    for point in palm_collision_points:
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
                        sphere.translate(point)
                        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
                        geometries.append(sphere)
                
                # FINGER COLLISION POINTS (magenta spheres)
                if len(finger_collisions) > 0 and np.any(finger_collisions):
                    finger_collision_points = all_potential_collision_points[finger_collisions]
                    for point in finger_collision_points:
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
                        sphere.translate(point)
                        sphere.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta
                        geometries.append(sphere)

            # 5. Cylinder visualization (small slab)
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius, 
                height=2 * half_height_cylinder,
                resolution=20
            )
            
            # Orient and position cylinder
            z_axis = np.array([0, 0, 1])
            if not np.allclose(lambda_dir, z_axis):
                if np.allclose(lambda_dir, -z_axis):
                    cyl_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
                else:
                    v = np.cross(z_axis, lambda_dir)
                    s = np.linalg.norm(v)
                    c = np.dot(z_axis, lambda_dir)
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    cyl_rotation = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
            else:
                cyl_rotation = np.eye(3)

            cylinder_transform = np.eye(4)
            cylinder_transform[:3, :3] = cyl_rotation
            cylinder_transform[:3, 3] = t_world
            cylinder.transform(cylinder_transform)

            # Create wireframe version
            cylinder_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder)
            cylinder_wireframe.paint_uniform_color([0.2, 0.2, 0.8])  # Blue wireframe
            geometries.append(cylinder_wireframe)

            # 6. Gripper geometry
            try:
                gripper_meshes = self.get_gripper_meshes(grasp_pose, gripper=G, color=(0.2, 0.8, 0.2))
                geometries.extend(gripper_meshes)
            except Exception as gripper_error:
                print(f"    [ERROR] Could not add gripper visualization: {gripper_error}")
            
            # 7. Main coordinate frame
            main_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            geometries.append(main_coord_frame)
            
            # Print legend
            print(f"TWO-SLAB COLLISION TEST VISUALIZATION:")
            print(f"  🟫 Gray:    Background points (outside large slab)")
            print(f"  🟢 Green:   Safe grasping points (inside cylinder)")
            print(f"  🟠 Orange:  Potential collision (outside cylinder in small slab)")
            print(f"  🟨 Blue:    Finger region points (large slab - small slab)")
            print(f"  🔴 Red:     PALM collision points")
            print(f"  🟣 Magenta: FINGER collision points")
            print(f"  🔵 Blue:    Collision cylinder (small slab)")
            print(f"  🤖 Robot:   Gripper geometry")
            print(f"{'-'*60}")
            print(f"  Small slab height: {2*half_height_cylinder:.3f}m")
            print(f"  Large slab height: {2*half_height_finger:.3f}m")
            print(f"  Cylinder radius: {radius:.3f}m")
            overall_result = "✅ NO COLLISION" if collision_result else "❌ COLLISION DETECTED"
            print(f"  OVERALL COLLISION TEST: {overall_result}")
            
            # Show visualization
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_name,
                zoom=0.6,
                front=[0, -1, 0],
                lookat=t_world,
                up=[0, 0, 1]
            )
            
        except Exception as e:
            print(f"Error in collision test visualization: {e}")
            import traceback
            traceback.print_exc()

    # =============================================================================
    # MAIN VISUALIZATION METHODS
    # =============================================================================
    
    def visualize_grasps(self, point_cloud_path: str = None, point_cloud_data: np.ndarray = None,
                        superquadric_params: Union[Dict, List[Dict]] = None, 
                        grasp_poses: Union[np.ndarray, List] = None,
                        show_sweep_volume: bool = False, gripper_colors: List = None,
                        window_name: str = "Superquadric Grasp Visualization",
                        align_finger_tips: bool = True, 
                        cad_model_path: str = None,
                        reference_model: o3d.geometry.PointCloud = None) -> None:
        """Main visualization method for superquadric grasps with smart view centering"""
        geometries = []
        
        # Load or create point cloud
        pcd = None
        if point_cloud_path is not None:
            pcd = self.load_point_cloud(point_cloud_path)
            if pcd is not None:
                pcd.paint_uniform_color([0.7, 0.7, 0.7])
                geometries.append(pcd)
        elif point_cloud_data is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
            geometries.append(pcd)
        
        # Add reference model (preferred over CAD model path)
        if reference_model is not None:
            reference_copy = o3d.geometry.PointCloud(reference_model)
            reference_copy.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for reference model
            geometries.append(reference_copy)
        # Add CAD model if provided and no reference model
        elif cad_model_path is not None:
            try:
                cad_mesh = o3d.io.read_triangle_mesh(cad_model_path)
                if cad_mesh.is_empty():
                    print(f"CAD model is empty: {cad_model_path}")
                else:
                    cad_mesh.compute_vertex_normals()
                    cad_mesh.paint_uniform_color([0.0, 0.0, 1.0])  # Blue color
                    geometries.append(cad_mesh)
            except Exception as e:
                print(f"Error loading CAD model: {e}")
        
        # Add superquadric meshes
        if superquadric_params is not None:
            if isinstance(superquadric_params, dict):
                superquadric_params = [superquadric_params]
            
            for i, sq_params in enumerate(superquadric_params):
                try:
                    sq_mesh = self.create_superquadric_mesh(
                        shape=sq_params['shape'],
                        scale=sq_params['scale'],
                        euler=sq_params['euler'],
                        translation=sq_params['translation']
                    )
                    
                    if sq_mesh is not None:
                        color = self.superquadric_colors[i % len(self.superquadric_colors)]
                        sq_mesh.paint_uniform_color(color)
                        geometries.append(sq_mesh)
                    
                except Exception as e:
                    print(f"Error creating superquadric {i+1}: {e}")
        
        # Add grasp visualizations
        if grasp_poses is not None:
            if not isinstance(grasp_poses, list):
                grasp_poses = [grasp_poses]
            
            # Default colors for grippers
            if gripper_colors is None:
                gripper_colors = self.gripper_colors
            
            for i, grasp_pose in enumerate(grasp_poses):
                try:
                    # Parse grasp pose to transformation matrix
                    transform_matrix = self.parse_grasp_pose(grasp_pose)
                    
                    # Get color for this gripper
                    color = gripper_colors[i % len(gripper_colors)]
                    
                    # Create gripper visualization
                    gripper_meshes = self.get_gripper_meshes(
                        transform_matrix,
                        show_sweep_volume=show_sweep_volume,
                        color=color,
                        finger_tip_to_origin=align_finger_tips
                    )
                    
                    geometries.extend(gripper_meshes)
                    
                    # Add coordinate frame at grasp pose
                    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
                    coord_frame.transform(transform_matrix)
                    geometries.append(coord_frame)
                    
                except Exception as e:
                    print(f"Error creating grasp visualization {i+1}: {e}")
        
        # Add small origin coordinate frame only if no grasps (to avoid clutter)
        main_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        geometries.append(main_coord_frame)
        
        # Calculate optimal view center and zoom
        view_center = self.get_view_center_priority(pcd, superquadric_params, grasp_poses)
        
        # Visualize all geometries with smart centering
        if len(geometries) > 0:       
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_name,
                front=[0, -1, 0],
                lookat=view_center,
                up=[0, 0, 1]
            )
        else:
            print("No geometries to visualize!")
           
    def get_view_center_priority(self, pcd, superquadric_params, grasp_poses):
        """Get view center with priority: point cloud > superquadrics > grasps > origin"""
        
        # Priority 1: Point cloud center
        if pcd is not None and len(pcd.points) > 0:
            return pcd.get_center()
        
        # Priority 2: Superquadric centers
        if superquadric_params:
            if isinstance(superquadric_params, dict):
                return np.array(superquadric_params['translation'])
            else:
                translations = [sq['translation'] for sq in superquadric_params]
                return np.mean(translations, axis=0)
        
        # Priority 3: Grasp centers
        if grasp_poses:
            if not isinstance(grasp_poses, list):
                grasp_poses = [grasp_poses]
            
            centers = []
            for grasp_pose in grasp_poses:
                transform = self.parse_grasp_pose(grasp_pose)
                centers.append(transform[:3, 3])
            
            if centers:
                return np.mean(centers, axis=0)
        
        # Priority 4: Origin as fallback
        return np.array([0, 0, 0])

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def load_point_cloud(self, file_path: str) -> Optional[o3d.geometry.PointCloud]:
        """Load point cloud from file"""
        try:
            if not os.path.exists(file_path):
                print(f"Point cloud file not found: {file_path}")
                return None
            
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                print(f"Point cloud file is empty: {file_path}")
                return None
            
            print(f"Loaded point cloud with {len(pcd.points)} points from {file_path}")
            return pcd
        
        except Exception as e:
            print(f"Error loading point cloud from {file_path}: {e}")
            return None
    
    def parse_grasp_pose(self, grasp_input: Union[np.ndarray, Dict, Tuple, List]) -> np.ndarray:
        """Parse grasp pose from various input formats"""
        try:
            if isinstance(grasp_input, np.ndarray) and grasp_input.shape == (4, 4):
                return grasp_input
            
            elif isinstance(grasp_input, dict):
                position = np.array(grasp_input['position'])
                if 'quaternion' in grasp_input:
                    quat = np.array(grasp_input['quaternion'])  # [x, y, z, w]
                    R = R_simple.from_quat(quat).as_matrix()
                elif 'rotation_matrix' in grasp_input:
                    R = np.array(grasp_input['rotation_matrix'])
                elif 'euler' in grasp_input:
                    euler = np.array(grasp_input['euler'])
                    R = R_simple.from_euler('xyz', euler).as_matrix()
                else:
                    R = np.eye(3)
                
                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = position
                return transform
            
            elif isinstance(grasp_input, (tuple, list)) and len(grasp_input) == 2:
                position, quat = grasp_input
                position = np.array(position)
                quat = np.array(quat)  # [x, y, z, w]
                R = R_simple.from_quat(quat).as_matrix()

                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = position
                return transform
            
            else:
                print("Invalid grasp pose format. Using identity matrix.")
                return np.eye(4)
        
        except Exception as e:
            print(f"Error parsing grasp pose: {e}")
            return np.eye(4)
        
    # =============================================================================
    # DEMO AND TESTING
    # =============================================================================
    
    def create_cubic_object(self, size: float = 0.05, center: Tuple[float, float, float] = (0, 0, 0), 
                          color: Tuple[float, float, float] = (0.8, 0.2, 0.2)) -> o3d.geometry.TriangleMesh:
        """Create a cubic object for testing/demo purposes"""
        cube = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        
        # Center the cube at the desired position
        cube.translate([-size/2, -size/2, -size/2])  # Center at origin first
        cube.translate(center)  # Then move to desired center
        
        cube.paint_uniform_color(color)
        cube.compute_vertex_normals()
        return cube

    def demo_visualization(self) -> None:
        """Demo showing basic visualization functionality"""
        # Create cube point cloud
        cube_points = []
        for x in np.linspace(-0.03, 0.03, 10):
            for y in np.linspace(-0.03, 0.03, 10):
                for z in np.linspace(-0.03, 0.03, 10):
                    if abs(x) > 0.025 or abs(y) > 0.025 or abs(z) > 0.025:
                        cube_points.append([x, y, z])
        
        cube_points = np.array(cube_points)
        
        # Define superquadric parameters (box-like)
        sq_params = {
            'shape': np.array([0.1, 0.1]),
            'scale': np.array([0.03, 0.03, 0.03]),
            'euler': np.array([0, 0, 0]),
            'translation': np.array([0, 0, 0])
        }
        
        # Example grasp
        grasp_pose = np.array([
            [0, 0, 1, -0.08],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        
        self.visualize_grasps(
            point_cloud_data=cube_points,
            superquadric_params=sq_params,
            grasp_poses=grasp_pose,
            show_sweep_volume=True,
            window_name="PerceptionVisualizer Demo"
        )

# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    # Run demo
    visualizer = PerceptionVisualizer()
    visualizer.demo_visualization()