#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class TargetPublisher(Node):
    def __init__(self):
        super().__init__('target_publisher')
        
        self.publisher = self.create_publisher(PoseStamped, '/target_pose', 10)
        
        # Parámetros configurables
        self.declare_parameter('distance_x', 0.4)
        self.declare_parameter('distance_y', 0.4)
        self.declare_parameter('threshold', 0.15)  # Umbral para considerar que llegó
        
        # Estado: 'phase_x' o 'phase_y' o 'completed'
        self.declare_parameter('current_phase', 'phase_x')
        self.phase = 'phase_x'
        
        # Timer para publicar a 10 Hz
        self.timer = self.create_timer(0.1, self.publish_target)
        
        self.get_logger().info('Target Publisher SECUENCIAL iniciado')
        self.get_logger().info('Fase: Primero X, luego Y')
    
    def publish_target(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        distance_x = self.get_parameter('distance_x').value
        distance_y = self.get_parameter('distance_y').value
        
        if self.phase == 'phase_x':
            # FASE 1: Solo movimiento en X
            msg.pose.position.x = distance_x
            msg.pose.position.y = 0.0
            self.get_logger().info(
                f'FASE X: Target a {distance_x}m adelante', 
                throttle_duration_sec=2.0
            )
            
        elif self.phase == 'phase_y':
            # FASE 2: Solo movimiento en Y (mantener X en 0)
            msg.pose.position.x = 0.0
            msg.pose.position.y = distance_y
            self.get_logger().info(
                f'FASE Y: Target a {distance_y}m lateral', 
                throttle_duration_sec=2.0
            )
            
        elif self.phase == 'completed':
            # COMPLETADO: Mantener en origen
            msg.pose.position.x = 0.0
            msg.pose.position.y = 0.0
            self.get_logger().info(
                'COMPLETADO: Robot en target final', 
                throttle_duration_sec=2.0
            )
        
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TargetPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()