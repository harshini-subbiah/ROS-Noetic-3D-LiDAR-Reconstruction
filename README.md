# 3D Semantic Environment Reconstruction using ROS Noetic

A ROS Noetic-based pipeline for reconstructing and visualizing a colored 3D environment from multi-sensor PR2 robot data. The project utilizes 2D LiDAR, IMU, wheel odometry, TF transformations, and stereo camera images recorded in a ROS bag to reconstruct a colored 3D environment and visualize it in RViz.

---

## Project Overview

This project demonstrates a complete robotics perception pipeline for reconstructing a 3D environment from recorded sensor data. Multiple onboard sensors are synchronized and processed to generate a colored point cloud that can be explored interactively in RViz.

The pipeline integrates:

- 2D LiDAR (`/base_scan`)
- Tilting LiDAR (`/tilt_scan`)
- Wheel Odometry
- IMU
- TF Transformations
- Stereo Camera Images

The final output is a reconstructed colored 3D map visualized using ROS RViz.

---

# Features

- Multi-sensor data synchronization
- Motion compensation using IMU and odometry
- 2D LiDAR to 3D point cloud reconstruction
- Stereo vision integration
- Dynamic object detection
- Semantic scene mapping
- Colored 3D environment visualization
- Interactive visualization in RViz
- Modular ROS-based pipeline

---

# Dataset

This project uses the **MIT CSAIL STATA Center PR2 Dataset**.

The dataset contains synchronized recordings from:

| Topic | Message Type |
|-------|--------------|
| `/base_scan` | sensor_msgs/LaserScan |
| `/tilt_scan` | sensor_msgs/LaserScan |
| `/base_odometry/odom` | nav_msgs/Odometry |
| `/robot_pose_ekf/odom_combined` | geometry_msgs/PoseWithCovarianceStamped |
| `/torso_lift_imu/data` | sensor_msgs/Imu |
| `/wide_stereo/left/image_rect` | sensor_msgs/Image |
| `/wide_stereo/right/image_rect` | sensor_msgs/Image |
| `/tf` | tf/tfMessage |

### Dataset Download

https://projects.csail.mit.edu/stata/downloads.html

> **Note:** The dataset is **not included** in this repository due to its size. Please download the ROS bag separately.

---

# Pipeline Architecture

<p align="center">
<img src="https://github.com/user-attachments/assets/33cad680-e59b-411e-8950-573b7ce16aa8" width="900">
</p>

---

# Technologies Used

- Ubuntu 20.04 LTS
- ROS Noetic
- Python 3
- RViz
- RTAB-Map
- OpenCV
- Open3D
- NumPy
- Matplotlib
- TF
- PCL
- Laser Assembler

---
# Quick Start

After completing the installation described in **setup.md**, run the project as follows.

### Terminal 1

```bash
roscore
```

### Terminal 2

```bash
source ~/.bashrc

source ~/ros_ws/devel/setup.bash

roslaunch mit_reconstruction reconstruct_3d.launch
```

If the launch file does **not** automatically play the ROS bag, open another terminal and run:

```bash
rosparam set use_sim_time true

rosbag play --clock <bag_file>.bag
```

The detailed explanation of each step and RViz configuration can be found in **setup.md**.

-----
# Output

The project generates:

- Colored 3D Point Cloud
- Semantic Environment Map
- Dynamic Object Visualization
- Interactive RViz Visualization
- Performance Graphs
- Generated Point Cloud Files

---

# Results

The reconstructed environment combines information from:

- Base LiDAR
- Tilting LiDAR
- Stereo Camera
- IMU
- Wheel Odometry
- TF Transformations

to generate a colored 3D representation of the surrounding environment.

---

# Acknowledgements

- MIT CSAIL STATA Center Dataset

---

# Author

**Harshini Subbiah**

---

# Contributors

Subiksha

----

---

# License

This project is licensed under the **MIT License**.
