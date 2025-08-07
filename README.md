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
- [Examples](#examples)
- [Controlling The Robot](#controlling-the-robot)
- [Troubleshooting](#troubleshooting)
- [Citations](#citations)
- [License & Contact](#license--contact)

---

## Prerequisites

- ROS2 (Humble/Iron recommended)
- Ubuntu 20.04/22.04
- Franka ROS2 workspace with [Franka ROS2](https://github.com/frankarobotics/franka_ros2) packages 
- [ZED SDK](https://www.stereolabs.com/en-ch/developers/release)

---

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

3. **Build the package**
   ```bash
   colcon build --packages-select superquadric_grasp_system
   source install/setup.bash
   ```

---

## Quick Start

### 1. Download Object Models
- [Download link – TBD](https://www.google.ch)

### 2. Set Up YOLO Segmentation
- Download pre-trained YOLOv11seg weights **or** train your own.
- Use the [YOLO-Finetune repo](https://github.com/MrGerencser/YOLO-Finetune) for:
  - Data collection

### 3. Set Camera Transforms
- Define camera transforms in:  
  `config/transformations.yaml`
- Use the [camera calibration tool](https://github.com/MrGerencser/camera_calibration) for gtting transformations.

### 4. Configure System Parameters
Edit the main perception config file at:  
`config/perception_config.yaml`

Key settings to check:
- **Camera settings:** serial numbers, resolution, transform file path
- **YOLO model path** and class IDs/names
- **Grasping method:** `"icp"` or `"superquadric"`
- **Point cloud filtering**, voxel sizes, workspace bounds
- **Visualization options**

### 5. Rebuild the Package
After changing any config files or models:
```bash
   cd ~/franka_ros2_ws
   colcon build --packages-select superquadric_grasp_system
   source install/setup.bash
   ```

### 6. Launch the Perception Node
```bash
# Terminal 1: Launch the system
ros2 run superquadric_grasp_system perception_node
```

### 7. Monitor Published Object Poses
```bash
# Terminal 2: Echo Poses
ros2 topic echo /perception/object_pose
```

---

## Examples
ICP example   
Superquadric example   

---

## Controlling The Robot

Once the perception node is running and object poses or grasp poses are being published, you can use the included executor script to send commands to the robot.

### Demo: Grasp Execution

This repository includes a [grasp_executor.py](superqaudric_grasp_system/grasp_executor.py) demo script for grasp execution which works with this [Cartesian Impedance Controller](https://github.com/MrGerencser/cartesian_impedance_control).   
In [grasp_executor.py](superqaudric_grasp_system/grasp_executor.py) you can specify wherevyou want to place the object by adjusting
```bash
'drop_box': {'x': 0.2, 'y': 0.6, 'z': 0.18}
```

Launch Cartesian Impdedance Controller
```bash
# Terminal 3: Launch Cartesian Impdedance Control
ros2 launch cartesian_impedance_control cartesian_impedance_controller.launch.py
```

Run Grasp Executor Node 
```bash
# Terminal 4: Run Grasp Executor Node
superquadric_grasp_system/grasp_executor.py
```

---

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

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue or contact [gejan@ethz.ch].
