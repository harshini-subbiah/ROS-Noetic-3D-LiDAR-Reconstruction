# 3D Semantic Environment Reconstruction using ROS Noetic

A ROS Noetic-based pipeline for reconstructing and visualizing a colored 3D environment from 2D LiDAR, IMU, odometry, TF, and stereo camera data recorded in a PR2 ROS bag. The project performs sensor synchronization, motion compensation, 3D point cloud generation, semantic segmentation, dynamic object detection, and interactive visualization in RViz.

---

## Project Overview

This project demonstrates a complete robotics perception pipeline that converts multi-sensor data from a recorded PR2 robot dataset into a colored 3D map for visualization and analysis.

The pipeline processes data from multiple sensors, synchronizes their timestamps, compensates for robot motion, reconstructs a 3D environment from 2D LiDAR scans, detects dynamic objects, applies semantic labeling, and visualizes the final colored point cloud in RViz.

---

## Features

- Multi-sensor data processing from ROS bag files
- Sensor synchronization
- Motion compensation using odometry and IMU
- 2D LiDAR to 3D point cloud reconstruction
- Dynamic object detection
- Semantic scene segmentation
- Colored 3D map generation
- Live RViz visualization
- Modular pipeline design

---

## Dataset

This project uses the **MIT CSAIL STATA Center PR2 Dataset**.

The dataset contains recorded ROS bag files with multiple synchronized sensors.

### Available Topics

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

> **Dataset Source:** https://projects.csail.mit.edu/stata/downloads.html

> **Note:** The ROS bag files are **not included** in this repository due to their large size. Please download the dataset from the official source above.

---

# Pipeline Architecture
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/33cad680-e59b-411e-8950-573b7ce16aa8" />

---

---

# Technologies Used

- ROS Noetic
- Ubuntu 20.04
- Python 3
- RViz
- OpenCV
- Open3D
- NumPy
- Matplotlib
- TF
- LaserScan
- PointCloud2

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/3D-Semantic-Environment-Reconstruction-ROS.git

cd 3D-Semantic-Environment-Reconstruction-ROS
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Source ROS Noetic:

```bash
source /opt/ros/noetic/setup.bash
```

---

# Running the Project

### 1. Start ROS Master

```bash
roscore
```

### 2. Enable Simulation Time

```bash
rosparam set use_sim_time true
```

### 3. Play the ROS Bag

```bash
rosbag play --clock <bag_file>.bag
```

### 4. Run the Processing Pipeline

```bash
python3 pipeline_runner.py
```

or

```bash
python3 scripts/09_live_rosbag_viz.py
```

(depending on your project setup)

### 5. Launch RViz

```bash
rosrun rviz rviz
```

### 6. Configure RViz

Set the **Fixed Frame** to:

```
world
```

or

```
map
```

depending on your visualization node.

Add the following displays:

- TF
- RobotModel
- PointCloud2
- LaserScan
- Marker
- MarkerArray
- Image
- Odometry

Select the published **PointCloud2** topic to visualize the reconstructed colored 3D environment.

---

# Output

The pipeline generates:

- Colored 3D point cloud
- Semantic environment map
- Dynamic object visualization
- Live RViz visualization
- Performance plots
- Generated point cloud files

---

# Results

The system reconstructs a colored 3D representation of the environment by combining data from:

- 2D LiDAR
- IMU
- Wheel Odometry
- TF Transformations
- Stereo Camera Images

The generated output can be explored interactively in RViz.

---

# Future Improvements

- Real-time SLAM integration
- Deep learning-based semantic segmentation
- GPU acceleration
- OctoMap integration
- Mesh reconstruction
- Multi-session mapping
- ROS 2 compatibility

---

---

# Acknowledgements

- MIT CSAIL STATA Center Dataset
---

# Author

**Harshini Subbiah**

---
# License

This project is licensed under the **MIT License**.

----
