from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # 1. Lanzar la base del LIMO
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('limo_base'),
                    'launch',
                    'limo_base.launch.py'
                ])
            ]),
            launch_arguments={'port_name': 'ttyUSB1'}.items()
        ),
        
        # 2. Nodo Person Tracker (con auto-select activado)
        Node(
            package='coordinates_generator',
            executable='person_tracker',
            name='person_tracker',
            output='screen',
            parameters=[
                {'auto_select_closest': True},  # Auto-seleccionar persona más cercana
                {'iou_threshold': 0.3},
                {'max_age': 30},
            ]
        ),
        
        # 3. Nodo Trajectory Controller
        Node(
            package='ejecucion',
            executable='trajectory_controller',
            name='trajectory_controller',
            output='screen',
            # parameters=[
            #     {'kp_linear': 0.5},
            #     {'kp_angular': 2.0},
            #     {'max_linear_speed': 0.2},
            #     {'max_angular_speed': 0.8},
            #     {'distance_threshold': 0.5},  # 50cm de distancia mínima
            # ]
        ),
    ])