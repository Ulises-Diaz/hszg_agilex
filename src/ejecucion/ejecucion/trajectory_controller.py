#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
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
        
        # Publisher de comandos
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Parámetros PID
        self.declare_parameter('kp_linear', 0.6)
        self.declare_parameter('ki_linear', 0.01)
        self.declare_parameter('kd_linear', 0.15)
        
        self.declare_parameter('kp_angular', 1.5)
        self.declare_parameter('ki_angular', 0.005)
        self.declare_parameter('kd_angular', 0.1)
        
        self.declare_parameter('max_linear_speed', 0.25)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('distance_threshold', 0.1)
        
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
        
        # Lista de waypoints (x, y) - MODIFICA AQUÍ TU TRAYECTORIA
        self.waypoints_rel = [
            [0.5, 0.0],   # Punto 1: 0.5m adelante
            [0.5, 0.5],   # Punto 2: 0.5m adelante, 0.5m derecha
            [0.0, 0.5],   # Punto 3: regresar en X, mantener Y
            [0.0, 0.0]    # Punto 4: volver al origen
        ]
        
        # Variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.start_x = None
        self.start_y = None
        self.current_waypoint_idx = 0
        self.waypoints_global = []
        self.initialized = False
        self.trajectory_completed = False
        
        self.last_time = time.time()
        
        # Timer para actualizar parámetros PID
        self.param_timer = self.create_timer(1.0, self.update_pid_parameters)
        
        self.get_logger().info('Trajectory Controller con PID completo iniciado')
        self.get_logger().info('Esperando odometría en /wheel/odom...')
    
    def update_pid_parameters(self):
        """Actualiza parámetros PID dinámicamente"""
        # PID Lineal
        self.pid_linear.Kp = self.get_parameter('kp_linear').value
        self.pid_linear.Ki = self.get_parameter('ki_linear').value
        self.pid_linear.Kd = self.get_parameter('kd_linear').value
        
        # PID Angular
        self.pid_angular.Kp = self.get_parameter('kp_angular').value
        self.pid_angular.Ki = self.get_parameter('ki_angular').value
        self.pid_angular.Kd = self.get_parameter('kd_angular').value
        
        # Actualizar límites
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
        """Callback de odometría"""
        # Actualizar posición actual
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        # Calcular dt
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Inicializar waypoints globales la primera vez
        if not self.initialized:
            self.start_x = self.current_x
            self.start_y = self.current_y
            
            # Convertir waypoints relativos a globales
            waypoints_rel = self.waypoints_rel
            for wp in waypoints_rel:
                # Transformar coordenadas relativas al marco global
                cos_yaw = math.cos(self.current_yaw)
                sin_yaw = math.sin(self.current_yaw)
                global_x = self.start_x + wp[0] * cos_yaw - wp[1] * sin_yaw
                global_y = self.start_y + wp[0] * sin_yaw + wp[1] * cos_yaw
                self.waypoints_global.append([global_x, global_y])
            
            self.initialized = True
            self.get_logger().info(f'Trayectoria iniciada con {len(self.waypoints_global)} waypoints')
            self.get_logger().info(f'Posición inicial: ({self.start_x:.2f}, {self.start_y:.2f}), Yaw: {math.degrees(self.current_yaw):.1f}°')
            for i, wp in enumerate(self.waypoints_global):
                self.get_logger().info(f'  Waypoint {i+1}: ({wp[0]:.2f}, {wp[1]:.2f})')
        
        # Si ya completó la trayectoria, detener
        if self.trajectory_completed:
            return
        
        # Verificar si hay waypoints pendientes
        if self.current_waypoint_idx >= len(self.waypoints_global):
            if not self.trajectory_completed:
                self.get_logger().info('🎉 ✅ TRAYECTORIA COMPLETADA!')
                self.trajectory_completed = True
                self.stop_robot()
                # Resetear PIDs
                self.pid_linear.reset()
                self.pid_angular.reset()
            return
        
        # Waypoint actual
        target_x = self.waypoints_global[self.current_waypoint_idx][0]
        target_y = self.waypoints_global[self.current_waypoint_idx][1]
        
        # Calcular error
        error_x = target_x - self.current_x
        error_y = target_y - self.current_y
        distance_to_waypoint = math.sqrt(error_x**2 + error_y**2)
        
        # Verificar si llegó al waypoint actual
        threshold = self.get_parameter('distance_threshold').value
        if distance_to_waypoint < threshold:
            self.get_logger().info(
                f'✅ Waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints_global)} alcanzado!'
            )
            self.current_waypoint_idx += 1
            
            # Resetear PIDs para el siguiente waypoint
            self.pid_linear.reset()
            self.pid_angular.reset()
            
            # Si hay más waypoints, continuar
            if self.current_waypoint_idx < len(self.waypoints_global):
                next_wp = self.waypoints_global[self.current_waypoint_idx]
                self.get_logger().info(
                    f'➡️  Siguiente waypoint: ({next_wp[0]:.2f}, {next_wp[1]:.2f})'
                )
            return
        
        # Calcular ángulo hacia el waypoint
        angle_to_waypoint = math.atan2(error_y, error_x)
        angle_error = angle_to_waypoint - self.current_yaw
        
        # Normalizar ángulo entre -pi y pi
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
        
        # Calcular velocidades con PID
        vel_linear = self.pid_linear.compute(distance_to_waypoint, dt)
        vel_angular = self.pid_angular.compute(angle_error, dt)
        
        # Publicar comando
        cmd = Twist()
        cmd.linear.x = vel_linear
        cmd.angular.z = vel_angular
        self.cmd_pub.publish(cmd)
        
        # Log
        self.get_logger().info(
            f'WP {self.current_waypoint_idx + 1}/{len(self.waypoints_global)} | '
            f'Pos: ({self.current_x:.2f}, {self.current_y:.2f}) | '
            f'Target: ({target_x:.2f}, {target_y:.2f}) | '
            f'Dist: {distance_to_waypoint:.2f}m | Ang: {math.degrees(angle_error):.1f}° | '
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
    node = TrajectoryController()
    
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