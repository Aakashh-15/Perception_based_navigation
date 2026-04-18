# Perception-based Navigation Stack

An advanced autonomous navigation system for the **R1 Mini** rover, designed for Gazebo/Ignition simulation. This project integrates the **Geometric Lookahead** algorithm for high-speed wall following with a **YOLOv8** vision pipeline for real-time object classification and sequential mission control.

---

## 📝 Project Overview

This repository implements a decoupled **"Split-Brain" architecture** to handle high-frequency control and heavy vision processing simultaneously:

- **Vision Core:** Manages computer vision tasks including YOLO-based logo classification and multi-dictionary ArUco/AprilTag detection (Omni-Scan).
- **Mission Controller:** A high-frequency control node that processes LiDAR data to execute smooth, predictive wall-following and state-dependent maneuvers (U-Turns, precision stopping).

---

## 🛠️ Requirements & Environment

| Component | Version |
|---|---|
| Operating System | Ubuntu 22.04 LTS (Jammy Jellyfish) |
| ROS 2 | Humble Hawksbill |
| Simulator | Gazebo / Ignition (Fortress or Garden) |

**Python Dependencies:**
- `ultralytics` (for YOLOv8)
- `opencv-python`
- `cv_bridge`

---

## 🚀 Getting Started

### 1. Installation & Build

Clone the repository and build the workspace:

```bash
git clone https://github.com/Aakashh-15/Perception_based_navigation.git
cd Perception_based_navigation
colcon build --symlink-install
source install/setup.bash
```

### 2. Launch Simulation

Open a terminal and launch the Gazebo simulation environment:

```bash
ros2 launch mini_r1_v1_gz sim.launch.py
```

### 3. Setup Hardware Bridge

In a **second terminal**, run the parameter bridge to synchronize ROS 2 with Ignition Gazebo (LiDAR, Camera, Clock, and Velocity commands):

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /r1_mini/lidar@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan \
  /cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist \
  /r1_mini/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image \
  /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock \
  --ros-args -r /r1_mini/lidar:=/scan
```

### 4. Run Vision Processing

In a **third terminal**, start the Vision Core (ensure you have sourced the setup file):

```bash
source install/setup.bash
ros2 run mini_r1_v1_gz vision_core.py
```

### 5. Start Mission Controller

In the **fourth terminal**, activate the control logic to begin autonomous navigation:

```bash
source install/setup.bash
ros2 run mini_r1_v1_gz mission_controller.py
```

---

## 🧠 Control Theory:

The robot utilizes a **predictive steering model** that projects a virtual point $L$ meters ahead of the robot. This allows the system to steer into corners *before* reaching them, eliminating the "ping-pong" oscillation effect.

### The Governing Formula

$$D_{t+1} = D_t + L \sin(\alpha)$$

| Variable | Description |
|---|---|
| $D_{t+1}$ | Predicted future distance to the wall |
| $L$ | Lookahead distance |
| $\alpha$ | Robot's heading angle relative to the wall (calculated via trigonometry) |

---

## 📊 Data Logging & Output

The system automatically generates two CSV files in the project root to track mission progress:

| File | Description |
|---|---|
| `logo_log.csv` | Stores timestamped classifications of floor logos (Normal, Green, Orange) |
| `aruco_log.csv` | Records detected waypoint IDs and their associated mission instructions |

---

## 📁 Repository Structure

```
Perception_based_navigation/
└── gazebo_gefier_r1-main/
    └── src/
        └── mini_r1_v1_gz/
            ├── vision_core.py
            ├── mission_controller.py
            └── ...
```

---

## 📄 License

This project is open-source. Feel free to fork, modify, and build upon it.

