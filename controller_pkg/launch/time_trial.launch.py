from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(
        package='controller_pkg', executable='time_trial', output='screen',
        parameters=[{'team': 'TeamName', 'password': 'pass', 'use_sim_time': True}])])
