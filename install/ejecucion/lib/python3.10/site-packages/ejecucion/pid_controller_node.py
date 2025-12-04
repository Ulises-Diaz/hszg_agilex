#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from ejecucion.pid_controller import PIDController
import time

class PIDControllerNode(Node):
    def __init__(self):
        super().__init__('pid_controller_node')
        
        # SUSCRIPCIÓN al tópico target_pose (ESTO FALTABA)
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self.pose_callback,
            10
        )
        
        # PUBLISHER a cmd_vel (ESTO FALTABA)
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Declarar parámetros configurables para PID lineal
        self.declare_parameter('kp_linear', 0.5)
        self.declare_parameter('ki_linear', 0.01)
        self.declare_parameter('kd_linear', 0.1)
        
        # Declarar parámetros configurables para PID angular
        self.declare_parameter('kp_angular', 1.0)
        self.declare_parameter('ki_angular', 0.01)
        self.declare_parameter('kd_angular', 0.05)
        
        # Declarar distancia objetivo
        self.declare_parameter('target_distance', 1.5)
        
        # Crear PIDs con valores de parámetros
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
            output_limits=(-0.8, 0.8)
        )
        
        # Usar parámetro para target_distance
        self.target_distance = self.get_parameter('target_distance').value
        
        # Variables de tiempo (ESTO FALTABA)
        self.last_time = time.time()
        self.last_msg_time = time.time()
        self.msg_timeout = 0.5
        
        # Timer para actualizar parámetros
        self.param_timer = self.create_timer(1.0, self.update_parameters)
        
        # Timer de seguridad (ESTO FALTABA)
        self.safety_timer = self.create_timer(1.0, self.safety_check)
        
        self.get_logger().info('PID Controller Node iniciado')
        self.get_logger().info('Esperando mensajes en /target_pose...')
    
    def update_parameters(self):
        """Actualiza los parámetros del PID dinámicamente"""
        # Actualizar constantes PID lineal
        self.pid_linear.Kp = self.get_parameter('kp_linear').value
        self.pid_linear.Ki = self.get_parameter('ki_linear').value
        self.pid_linear.Kd = self.get_parameter('kd_linear').value
        
        # Actualizar constantes PID angular
        self.pid_angular.Kp = self.get_parameter('kp_angular').value
        self.pid_angular.Ki = self.get_parameter('ki_angular').value
        self.pid_angular.Kd = self.get_parameter('kd_angular').value
        
        # Actualizar target distance
        self.target_distance = self.get_parameter('target_distance').value
    
    def pose_callback(self, msg: PoseStamped):
        """Callback cuando llega un mensaje de target_pose"""
        self.last_msg_time = time.time()
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        distance = msg.pose.position.x
        lateral_offset = msg.pose.position.y
        
        error_linear = distance - self.target_distance
        error_angular = -lateral_offset * 2.0
        
        vel_linear = self.pid_linear.compute(error_linear, dt)
        vel_angular = self.pid_angular.compute(error_angular, dt)
        
        cmd = Twist()
        cmd.linear.x = -vel_linear
        cmd.angular.z = vel_angular
        self.cmd_publisher.publish(cmd)  # AQUÍ FALTABA el "cmd"
        
        self.get_logger().info(
            f'Control - Dist: {distance:.2f}m (error: {error_linear:.2f}) | '
            f'Lateral: {lateral_offset:.2f}m | '
            f'Vel: lin={vel_linear:.2f}, ang={vel_angular:.2f}'
        )
    
    def safety_check(self):
        """Detiene el robot si no recibe mensajes"""
        time_since_msg = time.time() - self.last_msg_time
        
        if time_since_msg > self.msg_timeout:
            self.stop_robot()
            self.pid_linear.reset()
            self.pid_angular.reset()
    
    def stop_robot(self):
        """Detiene el robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PIDControllerNode()
    
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