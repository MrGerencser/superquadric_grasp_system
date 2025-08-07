# Superquadric Grasp System

*ROS 2 pipeline for robotic grasping with two interchangeable perception paths:*  
(1) **Model-based** – ICP alignment to known CAD / templates  
(2) **Learning-free, model-free** – **hidden superquadrics**  

*Optimised for **ZED** cameras and **Franka Emika Panda** arms.*

## Features
1. **Detect & segment** — YOLO-v11seg (use pre-trained models or your own)  
2. **Fuse & crop** — merge multi-view clouds, isolate each object  
3. **Estimate pose / grasps**  
   - **Model-based:** ICP → object pose (+ optional CAD grasps)  
   - **Model-free:** hidden superquadrics → generate antipodal grasp candidates → select best grasp pose based on filtering and scoring
4. **Plan & execute** — Cartesian-impedance demo for Franka Panda Emika Robot included  

<p align="center">
  <img src="resource/grasp_demo.gif" width="600" alt="Demo: grasping mugs, boxes and plush toys"/>
</p>

**Tested on:** ROS 2 Humble · Ubuntu 22.04 · ZED 2i (SDK 5.0.5) · Franka Panda Emika

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [ROS interfaces](#ros-interfaces)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Citations](#citations)
- [License & Contact](#license--contact)

---

## Prerequisites

- ROS2 (Humble/Iron recommended)
- Ubuntu 20.04/22.04
- Franka ROS2 workspace with [Franka ROS2](https://github.com/frankarobotics/franka_ros2) packages 
- [ZED SDK](https://www.stereolabs.com/en-ch/developers/release)

## Installation

1. **Clone the repository**
   ```bash
   cd ~/franka_ros2_ws/src
   git clone <repository-url> superquadric_grasp_system
   ```

2. **Install dependencies**
   ```bash
   cd ~/franka_ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. **Build the workspace**
   ```bash
   colcon build --packages-select superquadric_grasp_system
   source install/setup.bash
   ```

## Quick Start

1. Download Object models
2. Download yolo models or train your own. Check out this repo for recording some pictures for [YOLO-Finetune](https://github.com/MrGerencser/YOLO-Finetune).
3. Get correct camera transforms in config file. (Can be done by running this repo (camera calibration.

### Configuration

Edit the configuration file at `config/grasp_params.yaml` to adjust:
- Camera parameters
- Superquadric fitting thresholds
- Grasp generation settings
- Robot-specific parameters

### Running the Demo

```bash
# Terminal 1: Launch the system
ros2 run superquadric_grasp_system perception_node
```

1. **Detect & Segment** – YOLOv11seg (pre-trained or bring-your-own weights)
   - Get RGB/Depth with [ZED SDK](https://www.stereolabs.com/en-ch/developers/release).
   - Run YOLO object segmentation (pre-trained models can be downloaded here (link), train your own model following this repo [YOLO-Finetune](https://github.com/MrGerencser/YOLO-Finetune)).

2. **Fuse & Crop**
   - Fuse workspace point clouds from one or more cameras.
   - Crop out the per-object point cloud using the segmentation mask(s).

3. **Estimate Pose / Grasp**
   - **ICP path:** register object point cloud to a known model → object pose (and grasp pose if defined on the model). Usage example here.
   - **Superquadric path:** fit hidden superquadrics to the object cloud → generate antipodal grasp candidates → rank. Usage example here.

4. **Plan & Execute**
   - Publish grasp target → execute (this repo includes a demo [grasp_executor.py](superquadric_grasp_system/grasp_executor.py) that uses this [cartesian-impedance-controller]([https://github.com/MrGerencser/YOLO-Finetune](https://github.com/MrGerencser/cartesian_impedance_control)). 

### Basic Launch

```bash
ros2 run superquadric_grasp_system perception_node
```



# Terminal 2: Trigger grasp planning
ros2 service call /plan_grasp superquadric_grasp_system/srv/PlanGrasp "{target_object: 'cup'}"
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Point Cloud   │───▶│  Shape Fitting   │───▶│ Grasp Planning  │
│   Processing    │    │   (Superquadric) │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐            ▼
│   Execution     │◀───│   Motion         │    ┌─────────────────┐
│                 │    │   Planning       │◀───│ Pose Selection  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Nodes

- **`shape_fitter`**: Fits superquadric models to point clouds
- **`grasp_planner`**: Generates grasp poses from fitted shapes
- **`motion_executor`**: Executes planned motions on the robot

## Topics

- `/camera/depth/points` - Input point cloud
- `/fitted_shapes` - Visualization of fitted superquadrics
- `/grasp_poses` - Generated grasp candidates
- `/robot_state` - Current robot configuration

## Services

- `/plan_grasp` - Trigger grasp planning for specified object
- `/execute_grasp` - Execute the selected grasp
- `/reset_system` - Reset the planning pipeline

## Parameters

Key parameters in `config/grasp_params.yaml`:
- `fitting_tolerance`: Superquadric fitting accuracy (default: 0.01)
- `min_points`: Minimum points required for fitting (default: 100)
- `grasp_approach_distance`: Pre-grasp distance (default: 0.1m)
- `gripper_width`: Maximum gripper opening (default: 0.08m)

## Troubleshooting

### Common Issues

1. **No point cloud received**
   - Check camera connection and drivers
   - Verify topic names match configuration

2. **Poor shape fitting**
   - Adjust `fitting_tolerance` parameter
   - Ensure adequate lighting and object visibility

3. **Failed grasp execution**
   - Check robot safety limits
   - Verify collision detection settings

### Debug Mode

Run with debug visualization:
```bash
ros2 launch superquadric_grasp_system grasp_system.launch.py debug:=true
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue or contact [your-email@domain.com].
