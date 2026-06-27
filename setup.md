# Setup Guide

This project was developed using

- Ubuntu 20.04 LTS
- ROS Noetic
- Python 3.8
- RViz

---

# Step 1

Install ROS Noetic Desktop Full.

```bash
sudo apt update

sudo apt install ros-noetic-desktop-full
```

---

# Step 2

Initialize rosdep.

```bash
sudo rosdep init

rosdep update
```

---

# Step 3

Create a Catkin Workspace.

```bash
mkdir -p ~/ros_ws/src

cd ~/ros_ws

catkin_make
```

---

# Step 4

Source ROS.

```bash
source /opt/ros/noetic/setup.bash

source ~/ros_ws/devel/setup.bash
```

To make it permanent add it to ~/.bashrc.

```bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

echo "source ~/ros_ws/devel/setup.bash" >> ~/.bashrc

source ~/.bashrc
```

---

# Step 5

Clone the repository.

```bash
cd ~/ros_ws/src

git clone <repository-url>

cd ..

catkin_make
```

---

# Step 6

Install Python dependencies.

```bash
pip3 install -r requirements.txt
```

---

# Step 7

Download the MIT CSAIL PR2 Dataset.

https://projects.csail.mit.edu/stata/downloads.html

Place the downloaded .bag file anywhere on your system.

---

# Running the Project

## Terminal 1

Start ROS Master.

```bash
roscore
```

---

## Terminal 2

Source the workspace.

```bash
source ~/.bashrc

source ~/ros_ws/devel/setup.bash
```

Launch the reconstruction pipeline.

```bash
roslaunch mit_reconstruction reconstruct_3d.launch
```

This launch file:

- Starts RTAB-Map
- Starts Laser Assembler
- Configures TF
- Loads stereo image topics
- Loads odometry
- Loads LiDAR topics
- Opens RViz (if configured)

---

## (Optional) Terminal 3

If your launch file does **not** automatically play the rosbag, run:

```bash
rosparam set use_sim_time true

rosbag play --clock <bag_file>.bag
```

If the launch file already starts the rosbag playback, this step is **not required**.

---

## (Optional) Run Individual Pipeline Scripts

If you wish to execute the processing pipeline independently of the ROS launch file:

```bash
python3 pipeline_runner.py
```

or execute each module individually:

```bash
python3 scripts/00_bag_inspector.py

python3 scripts/01_synchronizer.py

python3 scripts/02_motion_compensator.py

python3 scripts/03_lidar_to_3d.py

python3 scripts/04_dynamic_detector.py

python3 scripts/05_semantic_segmentor.py

python3 scripts/07_map_builder.py

python3 scripts/08_visualizer.py

python3 scripts/09_live_rosbag_viz.py

python3 scripts/10_floor_separator.py

python3 scripts/generate_plots.py
```

Only execute these individually if they are designed to run as standalone scripts. Otherwise, use `pipeline_runner.py`.

---

# RViz Configuration

After RViz launches:

Set

```
Fixed Frame = map
```

or

```
world
```

depending on the published frame.

Add the following displays:

- TF
- PointCloud2
- MapCloud
- MapGraph
- LaserScan
- RobotModel
- Image
- Marker
- MarkerArray
- Odometry

Recommended topics:

```
PointCloud2

/cloud_map
```

```
MapCloud

/mapData
```

```
LaserScan

/base_scan
```

```
Image

/wide_stereo/left/image_rect_throttle
```

---

# Output

The pipeline reconstructs a colored 3D representation of the environment using:

- Base 2D LiDAR (`/base_scan`)
- Tilt LiDAR (`/tilt_scan`)
- IMU (`/torso_lift_imu/data`)
- Wheel Odometry (`/base_odometry/odom`)
- EKF Pose (`/robot_pose_ekf/odom_combined`)
- TF transformations (`/tf`)
- Stereo camera (`/wide_stereo`)

The reconstructed map is visualized interactively in RViz.

---

# Notes

- Ensure `use_sim_time` is enabled when replaying ROS bag files.
- If the rosbag finishes playing, restart the playback before checking ROS topics.
- The provided launch file is the recommended method for running the complete pipeline.
- Some RTAB-Map warnings related to scan synchronization may appear depending on the dataset timing and sensor rates. The project is designed to reconstruct the environment using the available synchronized sensor data.
