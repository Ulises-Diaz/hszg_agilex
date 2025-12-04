#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, String
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque


class PersonTrackerNode(Node):
    def __init__(self):
        super().__init__('person_tracker')
        
        # Parámetros
        self.declare_parameter('detections_topic', '/person_detections')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('target_pose_topic', '/target_person_pose')
        self.declare_parameter('tracker_debug_topic', '/tracker/debug_image')
        self.declare_parameter('select_target_topic', '/tracker/select_target')
        self.declare_parameter('tracker_status_topic', '/tracker/status')
        self.declare_parameter('iou_threshold', 0.3)  # Para matching
        self.declare_parameter('max_age', 30)  # Frames sin detección antes de eliminar
        self.declare_parameter('auto_select_closest', False)  # Auto-seleccionar persona más cercana
        
        detections_topic = self.get_parameter('detections_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        target_pose_topic = self.get_parameter('target_pose_topic').value
        debug_topic = self.get_parameter('tracker_debug_topic').value
        select_topic = self.get_parameter('select_target_topic').value
        status_topic = self.get_parameter('tracker_status_topic').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.max_age = self.get_parameter('max_age').value
        self.auto_select = self.get_parameter('auto_select_closest').value
        
        # OpenCV Bridge
        self.bridge = CvBridge()
        
        # Parámetros de cámara
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_received = False
        
        # Tracking
        self.tracks = {}  # {track_id: Track}
        self.next_id = 0
        self.selected_id = None
        self.latest_depth = None
        self.latest_rgb = None
        
        # Suscriptores
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            detections_topic,
            self.detections_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10
        )
        
        self.select_sub = self.create_subscription(
            Int32,
            select_topic,
            self.select_callback,
            10
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, target_pose_topic, 10)
        self.debug_pub = self.create_publisher(Image, debug_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        
        self.get_logger().info('Person Tracker iniciado')
        self.get_logger().info(f'Auto-select closest: {self.auto_select}')
        self.get_logger().info(f'Para seleccionar persona: ros2 topic pub {select_topic} std_msgs/msg/Int32 "data: ID"')
    
    def camera_info_callback(self, msg):
        """Obtener parámetros de cámara"""
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_info_received = True
            self.get_logger().info(
                f'Parámetros de cámara: fx={self.fx:.2f}, fy={self.fy:.2f}'
            )
    
    def depth_callback(self, msg):
        """Almacenar imagen de profundidad"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Error depth: {str(e)}')
    
    def select_callback(self, msg):
        """Seleccionar persona a seguir"""
        track_id = msg.data
        
        if track_id in self.tracks:
            self.selected_id = track_id
            self.get_logger().info(f'✓ Persona {track_id} seleccionada')
            status_msg = String()
            status_msg.data = f'tracking_id_{track_id}'
            self.status_pub.publish(status_msg)
        elif track_id == -1:
            self.selected_id = None
            self.get_logger().info('Tracking desactivado')
            status_msg = String()
            status_msg.data = 'no_target'
            self.status_pub.publish(status_msg)
        else:
            self.get_logger().warn(f'ID {track_id} no existe. IDs activos: {list(self.tracks.keys())}')
    
    def detections_callback(self, msg):
        """Procesar detecciones y hacer tracking"""
        try:
            current_detections = []
            
            # Convertir detecciones a formato simple
            for det in msg.detections:
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                w = det.bbox.size_x
                h = det.bbox.size_y
                conf = det.results[0].hypothesis.score
                
                bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                current_detections.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'center': (cx, cy)
                })
            
            # Actualizar tracks
            self.update_tracks(current_detections, msg.header)
            
            # Auto-seleccionar persona más cercana si está habilitado
            if self.auto_select and self.selected_id is None and len(self.tracks) > 0:
                self.auto_select_closest()
            
            # Publicar pose de persona seleccionada
            if self.selected_id is not None and self.selected_id in self.tracks:
                self.publish_target_pose(msg.header)
            
            # Publicar imagen de debug
            self.publish_debug_image()
            
        except Exception as e:
            self.get_logger().error(f'Error en detections_callback: {str(e)}')
    
    def update_tracks(self, detections, header):
        """Actualizar tracks con nuevas detecciones usando IoU"""
        # Marcar todos los tracks como no actualizados
        for track in self.tracks.values():
            track['updated'] = False
        
        # Matching con IoU
        matched_det_indices = set()
        
        for track_id, track in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = -1
            
            for idx, det in enumerate(detections):
                if idx in matched_det_indices:
                    continue
                
                iou = self.calculate_iou(track['bbox'], det['bbox'])
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = idx
            
            # Actualizar track si encontró match
            if best_det_idx >= 0:
                det = detections[best_det_idx]
                track['bbox'] = det['bbox']
                track['center'] = det['center']
                track['confidence'] = det['confidence']
                track['age'] = 0
                track['updated'] = True
                track['header'] = header
                matched_det_indices.add(best_det_idx)
            else:
                # No encontró match, incrementar edad
                track['age'] += 1
        
        # Crear nuevos tracks para detecciones no emparejadas
        for idx, det in enumerate(detections):
            if idx not in matched_det_indices:
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'bbox': det['bbox'],
                    'center': det['center'],
                    'confidence': det['confidence'],
                    'age': 0,
                    'updated': True,
                    'header': header
                }
                self.get_logger().info(f'Nueva persona detectada: ID {self.next_id}')
                self.next_id += 1
        
        # Eliminar tracks viejos
        to_remove = []
        for track_id, track in self.tracks.items():
            if track['age'] > self.max_age:
                to_remove.append(track_id)
                if track_id == self.selected_id:
                    self.get_logger().warn(f'Persona {track_id} perdida')
                    self.selected_id = None
        
        for track_id in to_remove:
            del self.tracks[track_id]
            self.get_logger().info(f'Track {track_id} eliminado (muy viejo)')
    
    def calculate_iou(self, bbox1, bbox2):
        """Calcular Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Área de intersección
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Área de unión
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def auto_select_closest(self):
        """Auto-seleccionar persona más cercana"""
        if not self.camera_info_received or self.latest_depth is None:
            return
        
        min_depth = float('inf')
        closest_id = None
        
        for track_id, track in self.tracks.items():
            cx, cy = track['center']
            depth = self.get_depth_at_point(int(cx), int(cy))
            
            if depth is not None and depth < min_depth:
                min_depth = depth
                closest_id = track_id
        
        if closest_id is not None:
            self.selected_id = closest_id
            self.get_logger().info(f'Auto-seleccionada persona más cercana: ID {closest_id} ({min_depth:.2f}m)')
    
    def get_depth_at_point(self, x, y):
        """Obtener profundidad en un punto"""
        if self.latest_depth is None:
            return None
        
        region_size = 10
        y_min = max(0, y - region_size)
        y_max = min(self.latest_depth.shape[0], y + region_size)
        x_min = max(0, x - region_size)
        x_max = min(self.latest_depth.shape[1], x + region_size)
        
        depth_region = self.latest_depth[y_min:y_max, x_min:x_max]
        valid_depths = depth_region[depth_region > 0]
        
        if len(valid_depths) > 0:
            depth = np.median(valid_depths)
            if depth > 100:  # Convertir mm a m
                depth = depth / 1000.0
            return float(depth)
        
        return None
    
    def publish_target_pose(self, header):
        """Publicar pose 3D de la persona seleccionada"""
        if not self.camera_info_received or self.latest_depth is None:
            return
        
        track = self.tracks[self.selected_id]
        cx, cy = track['center']
        
        depth = self.get_depth_at_point(int(cx), int(cy))
        
        if depth is not None:
            # Calcular posición 3D
            X = (cx - self.cx) * depth / self.fx
            Y = (cy - self.cy) * depth / self.fy
            Z = depth
            
            # Publicar PoseStamped
            pose_msg = PoseStamped()
            pose_msg.header = header
            pose_msg.pose.position.x = float(X)
            pose_msg.pose.position.y = float(Y)
            pose_msg.pose.position.z = float(Z)
            pose_msg.pose.orientation.w = 1.0
            
            self.pose_pub.publish(pose_msg)
            
            self.get_logger().info(
                f'Target ID {self.selected_id}: X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m'
            )
    
    def publish_debug_image(self):
        """Publicar imagen con todos los tracks"""
        if self.latest_depth is None:
            return
        
        # Crear imagen RGB desde depth para visualización
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(self.latest_depth, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        for track_id, track in self.tracks.items():
            x1, y1, x2, y2 = [int(v) for v in track['bbox']]
            cx, cy = [int(v) for v in track['center']]
            
            # Color: verde si está seleccionado, azul si no
            if track_id == self.selected_id:
                color = (0, 255, 0)  # Verde
                thickness = 3
            else:
                color = (255, 0, 0)  # Azul
                thickness = 2
            
            # Dibujar bbox
            cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(depth_colormap, (cx, cy), 5, color, -1)
            
            # Texto con ID
            text = f'ID:{track_id}'
            if track_id == self.selected_id:
                text += ' [TARGET]'
            
            cv2.putText(depth_colormap, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Info general
        info_text = f'Tracks: {len(self.tracks)} | Target: {self.selected_id if self.selected_id else "None"}'
        cv2.putText(depth_colormap, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Publicar
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(depth_colormap, 'bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Error debug: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()