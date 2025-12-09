#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32
from ejecucion.pid_controller import PIDController
import math
import time

class TrajectoryController(Node):
    def __init__(self):
        super().__init__('trajectory_controller')
        
        # Suscripciones
        self.odom_sub = self.create_subscription(Odometry, '/wheel/odom', self.odom_callback, 10)
        self.target_sub = self.create_subscription(PoseStamped, '/target_person_pose', self.target_pose_callback, 10)
        self.obstacle_sub = self.create_subscription(Bool, '/tracker/obstacle_detected', self.obstacle_callback, 10)
        self.obstacle_size_sub = self.create_subscription(Float32, '/tracker/obstacle_size', self.obstacle_size_callback, 10)
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Parámetros PID
        self.declare_parameter('kp_linear', 0.45)
        self.declare_parameter('ki_linear', 0.003)
        self.declare_parameter('kd_linear', 0.08)
        self.declare_parameter('kp_angular', 0.8)
        self.declare_parameter('ki_angular', 0.07)
        self.declare_parameter('kd_angular', 0.2)
        self.declare_parameter('max_linear_speed', 0.22)
        self.declare_parameter('max_angular_speed', 0.4)
        self.declare_parameter('distance_threshold', 0.3)
        self.declare_parameter('safety_distance', 0.2)
        
        # PARÁMETROS DEL ROBOT
        self.declare_parameter('robot_width', 1.12)  # ⭐ ANCHO DEL ROBOT (22cm)
        self.declare_parameter('robot_length', 0.7)  # Largo del robot (opcional)
        
        # Parámetros de evitación (calculados con ancho del robot)
        self.declare_parameter('obstacle_stop_distance', 0.4)
        self.declare_parameter('obstacle_avoidance_distance', 1.0)
        self.declare_parameter('lateral_clearance_multiplier', 1.5)  # ⭐ Clearance = ancho × 1.5
        self.declare_parameter('obstacle_slow_factor', 0.3)
        self.declare_parameter('small_obstacle_timeout', 2.5)
        self.declare_parameter('large_obstacle_timeout', 0.5)
        self.declare_parameter('avoidance_angular_speed', 0.25)
        self.declare_parameter('avoidance_linear_speed', 0.08)
        self.declare_parameter('large_obstacle_threshold', 0.6)
        
        # Crear PIDs
        self.pid_linear = PIDController(
            Kp=self.get_parameter('kp_linear').value,
            Ki=self.get_parameter('ki_linear').value,
            Kd=self.get_parameter('kd_linear').value,
            output_limits=(-self.get_parameter('max_linear_speed').value,
                          self.get_parameter('max_linear_speed').value)
        )
        
        self.pid_angular = PIDController(
            Kp=self.get_parameter('kp_angular').value,
            Ki=self.get_parameter('ki_angular').value,
            Kd=self.get_parameter('kd_angular').value,
            output_limits=(-self.get_parameter('max_angular_speed').value,
                          self.get_parameter('max_angular_speed').value)
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
        
        # Obstacle handling
        self.obstacle_detected = False
        self.obstacle_size = 0.0
        self.obstacle_stopped = False
        self.obstacle_stop_time = None
        self.avoiding_obstacle = False
        self.avoidance_direction = 1
        
        # Calcular clearance lateral basado en ancho del robot
        robot_width = self.get_parameter('robot_width').value
        clearance_mult = self.get_parameter('lateral_clearance_multiplier').value
        self.lateral_clearance = robot_width * clearance_mult  # ⭐ Ej: 0.22 × 1.5 = 0.33m
        
        self.last_time = time.time()
        
        self.param_timer = self.create_timer(1.0, self.update_pid_parameters)
        
        self.get_logger().info('🤖 Controller con GEOMETRÍA DEL ROBOT')
        self.get_logger().info(f'📐 Ancho robot: {robot_width}m')
        self.get_logger().info(f'📏 Clearance lateral: {self.lateral_clearance}m')
        self.get_logger().info(f'⚙️  PID Angular: Kp={self.get_parameter("kp_angular").value}')
        self.get_logger().info(f'🛣️  Distancia rodeo: {self.get_parameter("obstacle_avoidance_distance").value}m')
    
    def obstacle_callback(self, msg: Bool):
        """Callback para señal de obstáculo"""
        was_detected = self.obstacle_detected
        self.obstacle_detected = msg.data
        
        if not self.obstacle_detected and was_detected:
            self.get_logger().info('✅ Camino libre! Continuando...', throttle_duration_sec=1.0)
            self.obstacle_stopped = False
            self.obstacle_stop_time = None
            self.avoiding_obstacle = False
    
    def obstacle_size_callback(self, msg: Float32):
        """Callback para tamaño del obstáculo"""
        self.obstacle_size = msg.data
    
    def target_pose_callback(self, msg: PoseStamped):
        """Callback para target"""
        self.target_x = -msg.pose.position.x
        self.target_z = msg.pose.position.z
        self.has_target = True
        
        self.target_update_count += 1
        if self.target_update_count % 10 == 0:
            self.pid_linear.reset()
            self.pid_angular.reset()
    
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
        
        # Recalcular clearance si los parámetros cambiaron
        robot_width = self.get_parameter('robot_width').value
        clearance_mult = self.get_parameter('lateral_clearance_multiplier').value
        self.lateral_clearance = robot_width * clearance_mult
    
    def quaternion_to_yaw(self, q):
        """Convierte quaternion a yaw"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def calculate_avoidance_offset(self, lateral, is_large_obstacle):
        """
        Calcula el offset lateral necesario para rodear el obstáculo
        considerando el ancho del robot
        """
        # Si el obstáculo está al centro, necesitamos desplazarnos lateralmente
        # al menos: ancho_robot/2 + clearance
        
        robot_width = self.get_parameter('robot_width').value
        min_offset = robot_width / 2 + self.lateral_clearance  # ⭐ Ej: 0.11 + 0.33 = 0.44m
        
        # Si el obstáculo es grande (muro), necesitamos más offset
        if is_large_obstacle:
            min_offset *= 1.2  # 20% más para muros
        
        # Determinar hacia qué lado desviar
        if abs(lateral) < min_offset:
            # Obstáculo está muy al centro, necesitamos desviarnos
            # Elegir el lado con más espacio (hacia donde está la persona)
            direction = 1 if lateral >= 0 else -1
            target_offset = min_offset * direction
        else:
            # Obstáculo está lateral, ajustar un poco más
            target_offset = lateral + (min_offset * (1 if lateral > 0 else -1))
        
        return target_offset
    
    def odom_callback(self, msg: Odometry):
        """Callback de odometría - CON GEOMETRÍA DEL ROBOT"""
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
        # PARÁMETROS
        # ============================================================
        
        safety_distance = self.get_parameter('safety_distance').value
        obstacle_stop_dist = self.get_parameter('obstacle_stop_distance').value
        obstacle_avoid_dist = self.get_parameter('obstacle_avoidance_distance').value
        large_obs_thresh = self.get_parameter('large_obstacle_threshold').value
        robot_width = self.get_parameter('robot_width').value
        
        # Determinar si es obstáculo GRANDE (muro) o pequeño
        is_large_obstacle = self.obstacle_size > large_obs_thresh
        
        # Timeout adaptativo según tamaño
        if is_large_obstacle:
            obstacle_timeout = self.get_parameter('large_obstacle_timeout').value
        else:
            obstacle_timeout = self.get_parameter('small_obstacle_timeout').value
        
        # Coordenadas RELATIVAS
        lateral = self.target_x
        depth = self.target_z
        adjusted_depth = max(0.1, depth - safety_distance)
        
        # Calcular distancia al target
        distance_to_target = math.sqrt(lateral**2 + adjusted_depth**2)
        
        # Verificar si llegó
        threshold = self.get_parameter('distance_threshold').value
        if distance_to_target < threshold:
            self.get_logger().info('✅ Target alcanzado!')
            self.has_target = False
            self.stop_robot()
            self.pid_linear.reset()
            self.pid_angular.reset()
            self.obstacle_stopped = False
            self.obstacle_stop_time = None
            self.avoiding_obstacle = False
            return
        
        # ============================================================
        # AJUSTE DE DISTANCIAS CONSIDERANDO ANCHO DEL ROBOT
        # ============================================================
        
        # Ajustar distancia de stop considerando ancho del robot
        # Si el robot es más ancho, necesita detenerse más lejos
        adjusted_stop_dist = obstacle_stop_dist + (robot_width / 2)
        
        # Ajustar distancia de inicio de rodeo
        # Robots más anchos necesitan más espacio para maniobrar
        adjusted_avoid_dist = obstacle_avoid_dist + (robot_width / 2)
        
        # ============================================================
        # LÓGICA DE OBSTÁCULOS CON GEOMETRÍA
        # ============================================================
        
        if self.obstacle_detected:
            
            # CASO 1: MUY CERCA → DETENER
            if depth < adjusted_stop_dist:
                
                if not self.obstacle_stopped:
                    if is_large_obstacle:
                        self.get_logger().warn(
                            f'🛑 🧱 MURO a {depth:.2f}m (stop: {adjusted_stop_dist:.2f}m)!',
                            throttle_duration_sec=1.0
                        )
                    else:
                        self.get_logger().warn(
                            f'🛑 📦 OBSTÁCULO a {depth:.2f}m (stop: {adjusted_stop_dist:.2f}m)!',
                            throttle_duration_sec=1.0
                        )
                    
                    self.stop_robot()
                    self.obstacle_stopped = True
                    self.obstacle_stop_time = current_time
                    return
                
                # Ya detenido - verificar timeout
                time_stopped = current_time - self.obstacle_stop_time
                
                if time_stopped < obstacle_timeout:
                    remaining = obstacle_timeout - time_stopped
                    
                    if is_large_obstacle:
                        self.get_logger().info(
                            f'⏱️  🧱 Muro, rodeando en {remaining:.1f}s...',
                            throttle_duration_sec=0.5
                        )
                    else:
                        self.get_logger().info(
                            f'⏱️  📦 Esperando {remaining:.1f}s...',
                            throttle_duration_sec=0.5
                        )
                    
                    self.stop_robot()
                    return
                else:
                    # Timeout - activar modo rodeo
                    if not self.avoiding_obstacle:
                        self.avoiding_obstacle = True
                        # Calcular offset considerando ancho del robot
                        target_offset = self.calculate_avoidance_offset(lateral, is_large_obstacle)
                        self.avoidance_direction = 1 if target_offset > 0 else -1
                        
                        if is_large_obstacle:
                            self.get_logger().warn(
                                f'🔄 🧱 Rodeando muro (offset: {abs(target_offset):.2f}m, '
                                f'dir: {"derecha" if self.avoidance_direction > 0 else "izquierda"})!'
                            )
                        else:
                            self.get_logger().warn(
                                f'🔄 📦 Rodeando (offset: {abs(target_offset):.2f}m)!'
                            )
            
            # CASO 2: CERCA (zona de rodeo) → EMPEZAR A RODEAR
            elif depth < adjusted_avoid_dist:
                
                if not self.avoiding_obstacle:
                    self.avoiding_obstacle = True
                    target_offset = self.calculate_avoidance_offset(lateral, is_large_obstacle)
                    self.avoidance_direction = 1 if target_offset > 0 else -1
                    
                    if is_large_obstacle:
                        self.get_logger().warn(
                            f'↗️  🧱 Anticipando muro a {depth:.2f}m '
                            f'(inicio rodeo: {adjusted_avoid_dist:.2f}m, '
                            f'clearance: {self.lateral_clearance:.2f}m)',
                            throttle_duration_sec=1.0
                        )
                    else:
                        self.get_logger().warn(
                            f'↗️  📦 Anticipando obstáculo a {depth:.2f}m',
                            throttle_duration_sec=1.0
                        )
                
                # Resetear timer de stop si estaba activo
                self.obstacle_stopped = False
                self.obstacle_stop_time = None
            
            # CASO 3: LEJOS → REDUCIR VELOCIDAD PERO CONTINUAR
            else:
                self.avoiding_obstacle = False
                self.obstacle_stopped = False
        else:
            # Sin obstáculo
            self.avoiding_obstacle = False
            self.obstacle_stopped = False
            self.obstacle_stop_time = None
        
        # ============================================================
        # CONTROL DE MOVIMIENTO
        # ============================================================
        
        # Si está en MODO RODEO
        if self.avoiding_obstacle:
            avoidance_angular = self.get_parameter('avoidance_angular_speed').value
            avoidance_linear = self.get_parameter('avoidance_linear_speed').value
            
            # Calcular ángulo de rodeo considerando clearance necesario
            target_offset = self.calculate_avoidance_offset(lateral, is_large_obstacle)
            current_lateral_error = target_offset - lateral
            
            # Ajustar velocidad angular según cuánto necesitamos girar
            # Más error lateral = giro más rápido (pero limitado)
            angular_adjustment = min(abs(current_lateral_error) * 0.5, 1.0)
            vel_angular = avoidance_angular * self.avoidance_direction * angular_adjustment
            
            vel_linear = avoidance_linear
            
            # Si es muro, priorizar giro sobre avance
            if is_large_obstacle:
                vel_linear *= 0.5
                vel_angular *= 1.2
            
            cmd = Twist()
            cmd.linear.x = float(vel_linear)
            cmd.angular.z = float(vel_angular)
            self.cmd_pub.publish(cmd)
            
            obstacle_icon = "🧱" if is_large_obstacle else "📦"
            self.get_logger().info(
                f'🔄 {obstacle_icon} RODEANDO | Prof={depth:.2f}m | '
                f'Lat={lateral:.2f}m | Target offset={target_offset:.2f}m | '
                f'🚀 L:{vel_linear:.2f} A:{vel_angular:.2f}',
                throttle_duration_sec=0.5
            )
            return
        
        # CONTROL NORMAL (sin obstáculo o lejos)
        
        # Calcular ángulo RELATIVO
        angle_to_target = math.atan2(lateral, adjusted_depth)
        angle_error = angle_to_target
        
        # Factor de reducción según ángulo
        if abs(angle_error) > math.radians(30):
            linear_factor = 0.2
        elif abs(angle_error) > math.radians(15):
            linear_factor = 0.5
        else:
            linear_factor = 1.0
        
        # Si hay obstáculo lejos, reducir velocidad
        obstacle_factor = 1.0
        if self.obstacle_detected:
            obstacle_slow = self.get_parameter('obstacle_slow_factor').value
            obstacle_factor = obstacle_slow
        
        # Calcular velocidades con PID
        vel_linear = self.pid_linear.compute(adjusted_depth, dt) * linear_factor * obstacle_factor
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
        if self.obstacle_detected:
            if is_large_obstacle:
                obs_status = f" 🧱 {self.obstacle_size*100:.0f}%"
            else:
                obs_status = f" 📦 {self.obstacle_size*100:.0f}%"
        else:
            obs_status = ""
        
        self.get_logger().info(
            f'📍 Lat={lateral:.2f}m, Prof={adjusted_depth:.2f}m | '
            f'📏 {distance_to_target:.2f}m | 📐 {math.degrees(angle_error):.0f}° | '
            f'🚀 L:{vel_linear:.2f} A:{vel_angular:.2f}{obs_status}',
            throttle_duration_sec=0.5
        )
    
    def stop_robot(self):
        """Detiene el robot"""
        try:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
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