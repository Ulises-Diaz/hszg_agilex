#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from ejecucion.pid_controller import PIDController
import math
import time

class TrajectoryController(Node):
    def __init__(self):
        super().__init__('trajectory_controller')
        
        # Suscripción a odometría
        self.odom_sub = self.create_subscription(
            Odometry,
            '/wheel/odom',
            self.odom_callback,
            10
        )
        
        # Suscripción a target_pose
        self.target_sub = self.create_subscription(
            PoseStamped,
            '/target_person_pose',
            self.target_pose_callback,
            10
        )
        
        # Publisher de comandos
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Parámetros PID - OPTIMIZADOS
        self.declare_parameter('kp_linear', 0.45)
        self.declare_parameter('ki_linear', 0.003)
        self.declare_parameter('kd_linear', 0.08)
        
        self.declare_parameter('kp_angular', 0.6)  # Más bajo para suavidad
        self.declare_parameter('ki_angular', 0.001)
        self.declare_parameter('kd_angular', 0.05)
        
        self.declare_parameter('max_linear_speed', 0.22)
        self.declare_parameter('max_angular_speed', 0.4)  # Más bajo
        self.declare_parameter('distance_threshold', 0.3)
        self.declare_parameter('safety_distance', 0.2)
        
        # Crear PIDs
        self.pid_linear = PIDController(
            Kp=self.get_parameter('kp_linear').value,
            Ki=self.get_parameter('ki_linear').value,
            Kd=self.get_parameter('kd_linear').value,
            output_limits=(
                -self.get_parameter('max_linear_speed').value,
                self.get_parameter('max_linear_speed').value
            )
        )
        
        self.pid_angular = PIDController(
            Kp=self.get_parameter('kp_angular').value,
            Ki=self.get_parameter('ki_angular').value,
            Kd=self.get_parameter('kd_angular').value,
            output_limits=(
                -self.get_parameter('max_angular_speed').value,
                self.get_parameter('max_angular_speed').value
            )
        )
        
        # Variables
        self.target_x = None
        self.target_z = None
        self.has_target = False
        self.target_update_count = 0
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.initialized = False
        
        self.last_time = time.time()
        
        self.param_timer = self.create_timer(1.0, self.update_pid_parameters)
        
        self.get_logger().info('🤖 Controller SIMPLIFICADO - Control Reactivo')
        self.get_logger().info(f'⚙️  PID Angular: Kp={self.get_parameter("kp_angular").value}')
        self.get_logger().info(f'📏 Safety: {self.get_parameter("safety_distance").value}m')
    
    def target_pose_callback(self, msg: PoseStamped):
        """Callback para target - coordenadas relativas"""
        # Estas son RELATIVAS al robot/cámara
        self.target_x = -msg.pose.position.x  # Lateral
        self.target_z = msg.pose.position.z  # Profundidad
        self.has_target = True
        
        self.target_update_count += 1
        if self.target_update_count % 10 == 0:
            self.pid_linear.reset()
            self.pid_angular.reset()
        
        if self.target_update_count % 5 == 0:
            self.get_logger().info(
                f'🎯 Camera: X={self.target_x:.2f}m (lateral), Z={self.target_z:.2f}m (prof)',
                throttle_duration_sec=1.0
            )
    
    def update_pid_parameters(self):
        """Actualiza parámetros PID"""
        self.pid_linear.Kp = self.get_parameter('kp_linear').value
        self.pid_linear.Ki = self.get_parameter('ki_linear').value
        self.pid_linear.Kd = self.get_parameter('kd_linear').value
        
        self.pid_angular.Kp = self.get_parameter('kp_angular').value
        self.pid_angular.Ki = self.get_parameter('ki_angular').value
        self.pid_angular.Kd = self.get_parameter('kd_angular').value
        
        max_linear = self.get_parameter('max_linear_speed').value
        max_angular = self.get_parameter('max_angular_speed').value
        self.pid_linear.output_limits = (-max_linear, max_linear)
        self.pid_angular.output_limits = (-max_angular, max_angular)
    
    def quaternion_to_yaw(self, q):
        """Convierte quaternion a yaw"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def odom_callback(self, msg: Odometry):
        """Callback de odometría - CONTROL REACTIVO SIMPLE"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt < 0.001 or dt > 1.0:
            dt = 0.02
        
        if not self.initialized:
            self.initialized = True
            self.get_logger().info(
                f'✅ Odom: ({self.current_x:.2f}, {self.current_y:.2f}), Yaw: {math.degrees(self.current_yaw):.1f}°'
            )
        
        if not self.has_target:
            return
        
        # ============================================================
        # CONTROL REACTIVO SIMPLE - Usa coordenadas RELATIVAS directamente
        # ============================================================
        
        safety_distance = self.get_parameter('safety_distance').value
        
        # Coordenadas RELATIVAS al robot (desde la cámara)
        lateral = self.target_x      # Lateral: + derecha, - izquierda
        depth = self.target_z        # Profundidad: + adelante
        
        # Aplicar distancia de seguridad
        adjusted_depth = max(0.1, depth - safety_distance)
        
        # Calcular distancia euclidiana al target
        distance_to_target = math.sqrt(lateral**2 + adjusted_depth**2)
        
        # Verificar si llegó
        threshold = self.get_parameter('distance_threshold').value
        if distance_to_target < threshold:
            self.get_logger().info('✅ Target alcanzado!')
            self.has_target = False
            self.stop_robot()
            self.pid_linear.reset()
            self.pid_angular.reset()
            return
        
        # Calcular ángulo RELATIVO hacia el target
        # atan2(lateral, adelante) da el ángulo que necesitamos girar
        angle_to_target = math.atan2(lateral, adjusted_depth)
        
        # Este ángulo YA es el error angular (relativo al frente del robot)
        angle_error = angle_to_target
        
        # Factor de reducción según ángulo
        if abs(angle_error) > math.radians(30):
            linear_factor = 0.2  # Gira primero
        elif abs(angle_error) > math.radians(15):
            linear_factor = 0.5
        else:
            linear_factor = 1.0  # Avanza normalmente
        
        # Calcular velocidades con PID
        vel_linear = self.pid_linear.compute(adjusted_depth, dt) * linear_factor
        vel_angular = self.pid_angular.compute(angle_error, dt)
        
        # Reducir angular si estamos cerca
        if distance_to_target < 0.5:
            vel_angular *= 0.6
        
        # Publicar comando
        cmd = Twist()
        cmd.linear.x = float(vel_linear)
        cmd.angular.z = float(vel_angular)
        self.cmd_pub.publish(cmd)
        
        # Log
        self.get_logger().info(
            f'📍 Rel: Lat={lateral:.2f}m, Prof={adjusted_depth:.2f}m | '
            f'📏 Dist={distance_to_target:.2f}m | 📐 Ang={math.degrees(angle_error):.0f}° | '
            f'🚀 L:{vel_linear:.2f} A:{vel_angular:.2f}',
            throttle_duration_sec=0.5
        )
    
    def stop_robot(self):
        """Detiene el robot"""
        try:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.get_logger().info('🛑 Robot detenido')
        except Exception as e:
            self.get_logger().error(f'Error al detener: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Interrupción')
    except Exception as e:
        node.get_logger().error(f'Error: {e}')
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()