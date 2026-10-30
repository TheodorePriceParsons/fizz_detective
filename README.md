# Fizz Detective: ENPH 353 2026 competition

Our controller + training code. The sim repos (`2026_competition`, `my_robot_pkg`) are **not** in here; they sit next to this repo in the workspace.

## Workspace layout
```
~/2026_competition_ws/src/
├── 2026_competition/     # ENPH353 upstream (git pull to update)
├── my_robot_pkg/         # upstream URDF (fork it if we change the robot)
└── fizz_detective/       # this repo
    ├── controller_pkg/   # ROS 2 package that runs in competition
    └── training/         # data gen + CNN training (COLCON_IGNORE)
```

## Setup
```bash
cd ~/2026_competition_ws/src && git clone <this repo> fizz_detective
sudo apt install fonts-ubuntu          # needed so training data uses the sim's font
cd ~/2026_competition_ws && source /opt/ros/jazzy/setup.bash
colcon build --symlink-install && source install/setup.bash
```

## Run
```bash
# T1: sim
ros2 launch my_robot_bringup my_robot.launch.py
# T2: time trial (start timer, drive ~1.6 m, stop timer)
ros2 run controller_pkg time_trial --ros-args -p team:=TeamName -p password:=pass
```

## Rules we must not break
- From the sim we only **sub** `/camera/image_raw`, `/clock` and **pub** `/cmd_vel`, `/score_tracker`. No `/tf`, no NPC pose topics.
- ≥1 neural net we trained ourselves (YOLO is the only pretrained exception).
- RTF ≥ 0.5.
- Wait 1 s after creating publishers; don't spam `/score_tracker` (queue 1, keep ≥ 100 to 200 ms between messages).

## Training data
```bash
cd training && python3 -I gen_clue_data.py --n 5000
```
Renders banners exactly like `clueboard_generator.py` (same template, font, positions) and
crops characters on the fixed monospace grid → `data/chars/<C>/`.
