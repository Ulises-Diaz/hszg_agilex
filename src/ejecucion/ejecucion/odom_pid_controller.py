#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ejecucion.pid_controller import PIDController
import math
import time

class OdomPIDController(Node):
    def __init__(self):
        super().__init__('odom_pid_controller')
        
        # Suscripción a odometría
        self.odom_sub = self.create_subscription(
            Odometry,
            '/wheel/odom',
            self.odom_callback,
            10
        )
        
        # Publisher de comandos
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Parámetros configurables
        self.declare_parameter('target_x', 0.4)  # metros en X (adelante)
        self.declare_parameter('target_y', 0.2)  # metros en Y (lateral)
        self.declare_parameter('kp_linear', 0.5)
        self.declare_parameter('ki_linear', 0.01)
        self.declare_parameter('kd_linear', 0.1)
        self.declare_parameter('kp_angular', 2.0)
        self.declare_parameter('ki_angular', 0.01)
        self.declare_parameter('kd_angular', 0.1)
        self.declare_parameter('distance_threshold', 0.1)  # Umbral para considerar llegada
        
        # PIDs
        self.pid_linear = PIDController(
            Kp=self.get_parameter('kp_linear').value,
            Ki=self.get_parameter('ki_linear').value,
            Kd=self.get_parameter('kd_linear').value,
            output_limits=(-0.3, 0.3)
        )
        self.pid_angular = PIDController(
            Kp=self.get_parameter('kp_angular').value,
            Ki=self.get_parameter('ki_angular').value,
            Kd=self.get_parameter('kd_angular').value,
            output_limits=(-1.0, 1.0)
        )
        
        # Variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.target_x_global = None
        self.target_y_global = None
        self.initialized = False
        self.goal_reached = False
        
        self.last_time = time.time()
        
        # Timer para actualizar parámetros
        self.param_timer = self.create_timer(1.0, self.update_parameters)
        
        self.get_logger().info('Odom PID Controller iniciado')
        self.get_logger().info('Esperando odometría en /odom...')
    
    def update_parameters(self):
        """Actualiza parámetros dinámicamente"""
        self.pid_linear.Kp = self.get_parameter('kp_linear').value
        self.pid_linear.Ki = self.get_parameter('ki_linear').value
        self.pid_linear.Kd = self.get_parameter('kd_linear').value
        
        self.pid_angular.Kp = self.get_parameter('kp_angular').value
        self.pid_angular.Ki = self.get_parameter('ki_angular').value
        self.pid_angular.Kd = self.get_parameter('kd_angular').value
    
    def quaternion_to_yaw(self, q):
        """Convierte quaternion a yaw (ángulo en Z)"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def odom_callback(self, msg: Odometry):
        """Callback de odometría"""
        # Actualizar posición actual
        if not hasattr(self, 'callback_counter'):
            self.callback_counter = 0
        self.callback_counter += 1
        if self.callback_counter % 3 != 0:
            return 
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        # Inicializar target global la primera vez
        if not self.initialized:
            target_x_rel = self.get_parameter('target_x').value
            target_y_rel = self.get_parameter('target_y').value
            
            # Convertir target relativo a global
            self.target_x_global = self.current_x + target_x_rel * math.cos(self.current_yaw) - target_y_rel * math.sin(self.current_yaw)
            self.target_y_global = self.current_y + target_x_rel * math.sin(self.current_yaw) + target_y_rel * math.cos(self.current_yaw)
            
            self.initialized = True
            self.get_logger().info(f'Target establecido en: X={self.target_x_global:.2f}, Y={self.target_y_global:.2f}')
        
        # Calcular error
        error_x = self.target_x_global - self.current_x
        error_y = self.target_y_global - self.current_y
        
        # Distancia euclidiana al target
        distance_to_goal = math.sqrt(error_x**2 + error_y**2)
        
        # Ángulo hacia el target
        angle_to_goal = math.atan2(error_y, error_x)
        angle_error = angle_to_goal - self.current_yaw
        
        # Normalizar ángulo entre -pi y pi
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
        
        # Verificar si llegó al objetivo
        threshold = self.get_parameter('distance_threshold').value
        if distance_to_goal < threshold:
            if not self.goal_reached:
                self.get_logger().info('✅ TARGET ALCANZADO!')
                self.goal_reached = True
            self.stop_robot()
            return
        
        # Calcular dt
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Calcular velocidades con PID
        vel_linear = self.pid_linear.compute(distance_to_goal, dt)
        vel_angular = self.pid_angular.compute(angle_error, dt)
        
        # Publicar comandos
        cmd = Twist()
        cmd.linear.x = vel_linear
        cmd.angular.z = vel_angular
        self.cmd_pub.publish(cmd)
        
        # Log
        self.get_logger().info(
            f'Pos: ({self.current_x:.2f}, {self.current_y:.2f}) | '
            f'Target: ({self.target_x_global:.2f}, {self.target_y_global:.2f}) | '
            f'Dist: {distance_to_goal:.2f}m | Ang: {math.degrees(angle_error):.1f}° | '
            f'Vel: lin={vel_linear:.2f}, ang={vel_angular:.2f}',
            throttle_duration_sec=0.5
        )
    
    def stop_robot(self):
        """Detiene el robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = OdomPIDController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()