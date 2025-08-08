# Superquadric (Model-free) Example

## Overview
The Superquadric method fits a geometric primitive (superquadric) to the segmented point cloud.  
It does not require a CAD model and can generalize to new, unseen objects.

## Steps to Run

```bash
# Launch perception with Superquadric
ros2 run superquadric_grasp_system perception_node --ros-args -p grasping_method:=superquadric