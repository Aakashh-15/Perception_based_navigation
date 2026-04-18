import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_gz = get_package_share_directory('mini_r1_v1_gz')
    
    # Path to your bridge configuration
    bridge_config = os.path.join(pkg_gz, 'config', 'ros_gz_bridge.yaml')

    return LaunchDescription([
        # 1. Gazebo & Robot Spawn
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_gz, 'launch', 'sim.launch.py'))
        ),
        
        # 2. SLAM Toolbox (Asynchronous Mapping while navigating)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py'))
        ),

        # 3. ROS-Gazebo Bridge (Connects Camera, Lidar, and Cmd_Vel)
        # Node(
        #     package='ros_gz_bridge',
        #     executable='parameter_bridge',
        #     parameters=[{'config_file': bridge_config,
        #                  'use_sim_time': True}],
        #     output='screen'
        # ),

        # 4. YOLOv8 Logo & Red Tile Detector
        Node(
            package='mini_r1_v1_gz',
            executable='yolo_detector.py',
            output='screen',
            parameters=[{'use_sim_time': True}],
            remappings=[('/camera', '/r1_mini/camera/image_raw')]
        ),

        # 5. ArUco Detector
        # Node(
        #     package='aruco_opencv',
        #     executable='aruco_tracker',
        #     name='aruco_tracker',
        #     parameters=[{
        #         'image_topic': '/r1_mini/camera/image_raw',
        #         'camera_info_topic': '/r1_mini/camera/camera_info',
        #         'marker_dict': 'DICT_4X4_50',
        #         'image_transport': 'raw',
        #         'use_sim_time': True
        #     }]
            
        # ),

        # 5. ArUco Detector
        # 5. ArUco Detector
        Node(
            package='aruco_opencv',
            executable='aruco_tracker',
            name='aruco_tracker',
            parameters=[{
                'image_topic': '/r1_mini/camera/image_raw',
                'camera_info_topic': '/r1_mini/camera/camera_info',
                'marker_dict': '4X4_50', 
                'image_transport': 'raw',
                'use_sim_time': True
            }],
            remappings=[
                # Hackathon trick: Remap both possible default outputs to guarantee a connection!
                ('/aruco_tracker/aruco_detections', '/aruco_markers'),
                ('/aruco_detections', '/aruco_markers') 
            ]
        ),

        # 6. Mission Controller (The Brain)
        Node(
            package='mini_r1_v1_gz',
            executable='final_mission.py',
            output='screen',
            parameters=[{'use_sim_time': True}],
            remappings=[
                ('/scan', '/r1_mini/lidar'), # <--- FIXED: List of tuples []
                ('/odom', '/r1_mini/odom')
            ]
        )
    ])