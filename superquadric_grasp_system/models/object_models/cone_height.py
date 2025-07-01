import open3d as o3d
import numpy as np
import os

def analyze_cone_dimensions(ply_file_path):
    """Analyze cone.ply file and compute dimensions"""
    
    # Check if file exists
    if not os.path.exists(ply_file_path):
        print(f"Error: File {ply_file_path} not found")
        return None
    
    # Load the PLY file
    print(f"Loading: {ply_file_path}")
    mesh = o3d.io.read_triangle_mesh(ply_file_path)
    
    if len(mesh.vertices) == 0:
        print("Error: No vertices found in the mesh")
        return None
    
    # Convert to numpy array for easier computation
    vertices = np.asarray(mesh.vertices)
    
    # Compute bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    # Compute dimensions
    width_x = max_coords[0] - min_coords[0]
    width_y = max_coords[1] - min_coords[1]
    height_z = max_coords[2] - min_coords[2]
    
    # Compute center
    center = (min_coords + max_coords) / 2
    
    # Compute volume (approximate using bounding box)
    bounding_volume = width_x * width_y * height_z
    
    # For cone-specific analysis
    # Find base (lowest Z points) and tip (highest Z point)
    z_coords = vertices[:, 2]
    base_z = np.min(z_coords)
    tip_z = np.max(z_coords)
    cone_height = tip_z - base_z
    
    # Find base radius (points at lowest Z level)
    base_points = vertices[np.abs(z_coords - base_z) < 0.001]  # tolerance for base level
    if len(base_points) > 0:
        # Distance from center to farthest base point
        base_center = np.mean(base_points, axis=0)
        distances = np.linalg.norm(base_points - base_center, axis=1)
        base_radius = np.max(distances)
    else:
        base_radius = max(width_x, width_y) / 2
    
    # Print results
    print("\n=== CONE ANALYSIS RESULTS ===")
    print(f"File: {os.path.basename(ply_file_path)}")
    print(f"Number of vertices: {len(vertices)}")
    print(f"Number of triangles: {len(mesh.triangles)}")
    
    print(f"\nBounding Box:")
    print(f"  Min coordinates: [{min_coords[0]:.4f}, {min_coords[1]:.4f}, {min_coords[2]:.4f}]")
    print(f"  Max coordinates: [{max_coords[0]:.4f}, {max_coords[1]:.4f}, {max_coords[2]:.4f}]")
    
    print(f"\nDimensions:")
    print(f"  Width (X): {width_x:.4f} m")
    print(f"  Depth (Y): {width_y:.4f} m") 
    print(f"  Height (Z): {height_z:.4f} m")
    
    print(f"\nCone-specific:")
    print(f"  Cone height: {cone_height:.4f} m")
    print(f"  Base radius: {base_radius:.4f} m")
    print(f"  Base diameter: {base_radius * 2:.4f} m")
    
    print(f"\nCenter point: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"Bounding volume: {bounding_volume:.6f} m³")
    
    # Suggested grasp positions
    print(f"\n=== SUGGESTED GRASP POSITIONS ===")
    print(f"Side grasp 1: [{base_radius + 0.01:.4f}, 0.0000, {cone_height/3:.4f}]")
    print(f"Side grasp 2: [{-(base_radius + 0.01):.4f}, 0.0000, {cone_height/3:.4f}]")
    print(f"Top grasp:    [0.0000, 0.0000, {cone_height + 0.02:.4f}]")
    
    return {
        'width_x': width_x,
        'width_y': width_y,
        'height_z': height_z,
        'center': center,
        'cone_height': cone_height,
        'base_radius': base_radius,
        'min_coords': min_coords,
        'max_coords': max_coords,
        'bounding_volume': bounding_volume,
        'num_vertices': len(vertices),
        'num_triangles': len(mesh.triangles)
    }

def visualize_cone(ply_file_path):
    """Visualize the cone with coordinate frame"""
    mesh = o3d.io.read_triangle_mesh(ply_file_path)
    
    if len(mesh.vertices) == 0:
        print("Error: Cannot visualize empty mesh")
        return
    
    # Color the mesh
    mesh.paint_uniform_color([0.7, 0.3, 0.3])  # Reddish color
    mesh.compute_vertex_normals()
    
    # Create coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    
    # Visualize
    print("Visualizing cone... Press Q to quit")
    o3d.visualization.draw_geometries([mesh, coord_frame])

if __name__ == "__main__":
    # Path to cone.ply file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cone_ply_path = os.path.join(current_dir, "Cone.ply")
    
    # Alternative paths if not found
    if not os.path.exists(cone_ply_path):
        cone_ply_path = os.path.join(current_dir, "cone.ply")
    
    if not os.path.exists(cone_ply_path):
        print("Searching for cone PLY file...")
        for root, dirs, files in os.walk(current_dir):
            for file in files:
                if 'cone' in file.lower() and file.endswith('.ply'):
                    cone_ply_path = os.path.join(root, file)
                    print(f"Found: {cone_ply_path}")
                    break
    
    # Analyze the cone
    results = analyze_cone_dimensions(cone_ply_path)
    
    # Optional: visualize the cone
    visualize_choice = input("\nVisualize cone? (y/n): ").lower().strip()
    if visualize_choice == 'y':
        visualize_cone(cone_ply_path)
    
    if results:
        print(f"\nAnalysis complete! Results saved in memory.")
        
        # Generate grasp YAML suggestion
        print(f"\n=== GENERATED GRASP YAML ===")
        print("grasps:")
        print(f"  - name: \"side_grasp_1\"")
        print(f"    position: [{results['base_radius'] + 0.015:.4f}, 0.0000, {results['cone_height']/3:.4f}]")
        print(f"    orientation: [0.0, 1.57, 0.0]")
        print(f"  - name: \"side_grasp_2\"")
        print(f"    position: [{-(results['base_radius'] + 0.015):.4f}, 0.0000, {results['cone_height']/3:.4f}]")
        print(f"    orientation: [0.0, -1.57, 0.0]")
        print(f"  - name: \"top_grasp\"")
        print(f"    position: [0.0000, 0.0000, {results['cone_height'] + 0.025:.4f}]")
        print(f"    orientation: [0.0, 0.0, 0.0]")