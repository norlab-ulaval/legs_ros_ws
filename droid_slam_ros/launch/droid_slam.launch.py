from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('droid_slam_ros')
    
    # Default weights path
    default_weights = os.path.join(pkg_share, 'droid.pth')

    # Config file
    config_file = os.path.join(pkg_share, 'config', 'droid_slam.yaml')

    return LaunchDescription([
        # Keep weights argument for easy override
        DeclareLaunchArgument('weights', default_value=default_weights, description='Path to model weights'),
        
        Node(
            package='droid_slam_ros',
            executable='ros_node.py',
            name='droid_node',
            namespace=os.getenv('NAMESPACE', ''),
            output="screen",
            sigterm_timeout="60",  # Wait 60 seconds before escalating to SIGTERM
            sigkill_timeout="10",  # Wait 10 more seconds before SIGKILL
            parameters=[
                config_file,
                {'weights': LaunchConfiguration('weights')}, # Override weights from launch arg
                {'stereo': True} # Enforce stereo
            ]
        )
    ])
