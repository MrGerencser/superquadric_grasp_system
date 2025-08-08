# ICP (Model-based) Example

## Overview
The ICP (Iterative Closest Point) method aligns a segmented point cloud to a known CAD model or template.  
It provides accurate 6D pose estimation when the object geometry is known in advance.

## Steps to Run

```bash
# Launch perception with ICP
ros2 run superquadric_grasp_system perception_node --ros-args -p grasping_method:=icp
