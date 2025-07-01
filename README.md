# Superquadric Grasp System

A ROS2-based robotic grasping system that uses superquadric shape fitting for intelligent object manipulation with Franka Emika robots.

## Overview

This package implements a sophisticated grasping pipeline that:
- Fits superquadric shapes to point cloud data
- Generates optimal grasp poses based on object geometry
- Provides robust manipulation capabilities for various object shapes
- Integrates seamlessly with Franka Emika robotic arms

## Prerequisites

- ROS2 (Humble/Iron recommended)
- Ubuntu 20.04/22.04
- Franka ROS2 packages

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

## Usage

### Basic Launch

```bash
ros2 run superquadric_grasp_system perception_node
```

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

## Citation

If you use this work in your research, please cite:
```
@article{superquadric_grasp_2024,
  title={Superquadric-Based Grasp Planning for Robotic Manipulation},
  author={Your Name},
  journal={Robotics Research},
  year={2024}
}
```

## Contact

For questions and support, please open an issue or contact [your-email@domain.com].
