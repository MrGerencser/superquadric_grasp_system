#!/usr/bin/env python3
"""
webvis_demo.py – view a point cloud and cube in the desktop window
and at http://localhost:8888 via WebRTC.
"""

import numpy as np, open3d as o3d
from open3d.visualization import draw

# 1) turn the WebRTC server on  – one-liner!
o3d.visualization.webrtc_server.enable_webrtc()             # streams every GUI window :contentReference[oaicite:4]{index=4}

# 2) demo geometries ---------------------------------------------------------
pcd = o3d.io.read_point_cloud(o3d.data.PCDPointCloud().path)  # built-in sample, downloads once :contentReference[oaicite:5]{index=5}
pcd = pcd.random_down_sample(0.05)
pcd.paint_uniform_color([0.1, 0.8, 1.0])

cube = o3d.geometry.TriangleMesh.create_box()
cube.compute_vertex_normals()                                 # lit meshes need normals :contentReference[oaicite:6]{index=6}
cube.translate([2, 0, 0])
cube.paint_uniform_color([1.0, 0.3, 0.3])

# 3) visualize with friendly names ------------------------------------------
draw(
    [
        {"name": "Point Cloud", "geometry": pcd},
        {"name": "Red Cube",    "geometry": cube}
    ],
    title="Open3D WebVisualizer demo",
    show_ui=True
)
