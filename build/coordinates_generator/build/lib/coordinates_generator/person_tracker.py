#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, String, Bool, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque


class PersonTrackerNode(Node):
    def __init__(self):
        super().__init__('person_tracker')
        
        # Parámetros existentes
        self.declare_parameter('detections_topic', '/person_detections')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('target_pose_topic', '/target_person_pose')
        self.declare_parameter('tracker_debug_topic', '/tracker/debug_image')
        self.declare_parameter('select_target_topic', '/tracker/select_target')
        self.declare_parameter('tracker_status_topic', '/tracker/status')
        self.declare_parameter('obstacle_detected_topic', '/tracker/obstacle_detected')
        self.declare_parameter('obstacle_size_topic', '/tracker/obstacle_size')
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('max_age', 30)
        self.declare_parameter('auto_select_closest', False)
        
        # PARÁMETROS DE GEOMETRÍA DEL ROBOT
        self.declare_parameter('robot_width', 0.22)
        self.declare_parameter('robot_length', 0.45)
        
        # Parámetros de detección de obstáculos
        self.declare_parameter('enable_obstacle_detection', True)
        self.declare_parameter('obstacle_min_distance', 0.5)
        self.declare_parameter('obstacle_width_threshold', 0.3)
        self.declare_parameter('path_check_width_multiplier', 1.8)
        self.declare_parameter('large_obstacle_threshold', 0.6)
        self.declare_parameter('obstacle_depth_override', True)  # ⭐ NUEVO: Usar profundidad del obstáculo
        
        detections_topic = self.get_parameter('detections_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        target_pose_topic = self.get_parameter('target_pose_topic').value
        debug_topic = self.get_parameter('tracker_debug_topic').value
        select_topic = self.get_parameter('select_target_topic').value
        status_topic = self.get_parameter('tracker_status_topic').value
        obstacle_topic = self.get_parameter('obstacle_detected_topic').value
        obstacle_size_topic = self.get_parameter('obstacle_size_topic').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.max_age = self.get_parameter('max_age').value
        self.auto_select = self.get_parameter('auto_select_closest').value
        
        self.enable_obstacle = self.get_parameter('enable_obstacle_detection').value
        self.obstacle_min_dist = self.get_parameter('obstacle_min_distance').value
        self.obstacle_width_thresh = self.get_parameter('obstacle_width_threshold').value
        self.large_obstacle_thresh = self.get_parameter('large_obstacle_threshold').value
        self.use_obstacle_depth = self.get_parameter('obstacle_depth_override').value  # ⭐
        
        # Calcular path_width basado en geometría del robot
        robot_width = self.get_parameter('robot_width').value
        path_multiplier = self.get_parameter('path_check_width_multiplier').value
        self.path_width = robot_width * path_multiplier
        
        # OpenCV Bridge
        self.bridge = CvBridge()
        
        # Parámetros de cámara
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_received = False
        
        # Tracking
        self.tracks = {}
        self.next_id = 0
        self.selected_id = None
        self.latest_depth = None
        self.latest_rgb = None
        
        # Obstacle detection
        self.obstacle_detected = False
        self.obstacle_side = "none"
        self.obstacle_coverage = 0.0
        self.obstacle_min_depth = None  # ⭐ NUEVO: Profundidad mínima del obstáculo
        self.obstacle_lateral_pos = 0.0  # ⭐ NUEVO: Posición lateral del obstáculo
        
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
        self.obstacle_pub = self.create_publisher(Bool, obstacle_topic, 10)
        self.obstacle_size_pub = self.create_publisher(Float32, obstacle_size_topic, 10)
        
        self.get_logger().info('🤖 Person Tracker con AJUSTE DE PROFUNDIDAD')
        self.get_logger().info(f'Auto-select closest: {self.auto_select}')
        self.get_logger().info(f'📐 Ancho robot: {robot_width}m')
        self.get_logger().info(f'🛣️  Ancho camino verificado: {self.path_width:.2f}m (robot × {path_multiplier})')
        self.get_logger().info(f'⚠️  Obstacle detection: {self.enable_obstacle}')
        self.get_logger().info(f'🎯 Usar profundidad de obstáculo: {self.use_obstacle_depth}')
    
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
            
            # Auto-seleccionar persona más cercana
            if self.auto_select and self.selected_id is None and len(self.tracks) > 0:
                self.auto_select_closest()
            
            # Detectar obstáculos en el camino
            if self.selected_id is not None and self.selected_id in self.tracks:
                if self.enable_obstacle:
                    self.detect_obstacles_in_path()
                
                # Publicar pose de persona seleccionada
                self.publish_target_pose(msg.header)
            
            # Publicar imagen de debug
            self.publish_debug_image()
            
        except Exception as e:
            self.get_logger().error(f'Error en detections_callback: {str(e)}')
    
    def detect_obstacles_in_path(self):
        """Detectar obstáculos y guardar su profundidad mínima"""
        if not self.camera_info_received or self.latest_depth is None:
            return
        
        track = self.tracks[self.selected_id]
        cx, cy = track['center']
        
        # Obtener profundidad de la persona
        person_depth = self.get_depth_at_point(int(cx), int(cy))
        if person_depth is None:
            return
        
        # Definir región del camino a verificar
        img_center_x = self.cx
        img_height, img_width = self.latest_depth.shape
        
        path_width_pixels = int((self.path_width * self.fx) / 1.0)
        
        # Región vertical del camino (mitad inferior de la imagen)
        path_y_start = int(img_height * 0.5)
        path_y_end = img_height
        
        # Región horizontal del camino (centrada)
        path_x_start = int(img_center_x - path_width_pixels / 2)
        path_x_end = int(img_center_x + path_width_pixels / 2)
        
        # Asegurar límites
        path_x_start = max(0, path_x_start)
        path_x_end = min(img_width, path_x_end)
        path_y_start = max(0, path_y_start)
        path_y_end = min(img_height, path_y_end)
        
        # Extraer región del camino
        path_region = self.latest_depth[path_y_start:path_y_end, path_x_start:path_x_end]
        
        # Filtrar profundidades válidas (más cerca que la persona)
        valid_depths = path_region[(path_region > 0) & (path_region < person_depth * 1000)]  # mm
        
        if len(valid_depths) > 0:
            # Convertir a metros si es necesario
            valid_depths = valid_depths / 1000.0 if valid_depths.max() > 100 else valid_depths
            
            # Encontrar obstáculos cercanos
            close_obstacles = valid_depths[valid_depths < self.obstacle_min_dist]
            
            # Calcular % del área bloqueada
            self.obstacle_coverage = len(close_obstacles) / path_region.size if path_region.size > 0 else 0
            
            # Umbral mínimo: 5% del área tiene obstáculos
            if len(close_obstacles) > path_region.size * 0.05:
                self.obstacle_detected = True
                
                # ⭐ NUEVO: Guardar profundidad mínima del obstáculo
                self.obstacle_min_depth = float(np.min(close_obstacles))
                
                # ⭐ NUEVO: Calcular posición lateral del obstáculo
                # Encontrar píxeles del obstáculo en la imagen
                obstacle_mask = (path_region > 0) & (path_region < self.obstacle_min_dist * 1000)
                if np.any(obstacle_mask):
                    y_coords, x_coords = np.where(obstacle_mask)
                    # Centro horizontal del obstáculo (en píxeles del path_region)
                    obstacle_center_x_px = int(np.mean(x_coords))
                    # Convertir a coordenadas de imagen completa
                    obstacle_img_x = path_x_start + obstacle_center_x_px
                    # Convertir a coordenadas laterales (metros)
                    self.obstacle_lateral_pos = (obstacle_img_x - self.cx) * self.obstacle_min_depth / self.fx
                else:
                    self.obstacle_lateral_pos = 0.0
                
                # Determinar en qué lado está el obstáculo
                left_half = path_region[:, :path_region.shape[1]//2]
                right_half = path_region[:, path_region.shape[1]//2:]
                
                left_obstacles = np.sum((left_half > 0) & (left_half < self.obstacle_min_dist * 1000))
                right_obstacles = np.sum((right_half > 0) & (right_half < self.obstacle_min_dist * 1000))
                
                if left_obstacles > right_obstacles * 1.5:
                    self.obstacle_side = "left"
                elif right_obstacles > left_obstacles * 1.5:
                    self.obstacle_side = "right"
                else:
                    self.obstacle_side = "center"
                
                # Publicar señal de obstáculo
                obstacle_msg = Bool()
                obstacle_msg.data = True
                self.obstacle_pub.publish(obstacle_msg)
                
                # Publicar tamaño del obstáculo (0.0 a 1.0)
                size_msg = Float32()
                size_msg.data = float(self.obstacle_coverage)
                self.obstacle_size_pub.publish(size_msg)
                
                # Determinar tipo de obstáculo
                if self.obstacle_coverage > self.large_obstacle_thresh:
                    obstacle_type = "MURO/GRANDE"
                    icon = "🧱"
                else:
                    obstacle_type = "PEQUEÑO"
                    icon = "📦"
                
                self.get_logger().warn(
                    f'⚠️  {icon} OBSTÁCULO {obstacle_type} ({self.obstacle_coverage*100:.0f}%) - '
                    f'Lado: {self.obstacle_side} | Dist: {self.obstacle_min_depth:.2f}m | '
                    f'Lat: {self.obstacle_lateral_pos:.2f}m',
                    throttle_duration_sec=1.0
                )
            else:
                self.obstacle_detected = False
                self.obstacle_side = "none"
                self.obstacle_coverage = 0.0
                self.obstacle_min_depth = None
                self.obstacle_lateral_pos = 0.0
                
                # Publicar señal de camino libre
                obstacle_msg = Bool()
                obstacle_msg.data = False
                self.obstacle_pub.publish(obstacle_msg)
                
                size_msg = Float32()
                size_msg.data = 0.0
                self.obstacle_size_pub.publish(size_msg)
        else:
            self.obstacle_detected = False
            self.obstacle_side = "none"
            self.obstacle_coverage = 0.0
            self.obstacle_min_depth = None
            self.obstacle_lateral_pos = 0.0
    
    def update_tracks(self, detections, header):
        """Actualizar tracks con nuevas detecciones usando IoU"""
        for track in self.tracks.values():
            track['updated'] = False
        
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
                track['age'] += 1
        
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
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
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
            if depth > 100:
                depth = depth / 1000.0
            return float(depth)
        
        return None
    
    def publish_target_pose(self, header):
        """Publicar pose 3D con ajuste de profundidad si hay obstáculo"""
        if not self.camera_info_received or self.latest_depth is None:
            return
        
        track = self.tracks[self.selected_id]
        cx, cy = track['center']
        
        depth = self.get_depth_at_point(int(cx), int(cy))
        
        if depth is not None:
            X = (cx - self.cx) * depth / self.fx
            Y = (cy - self.cy) * depth / self.fy
            Z = depth
            
            # ⭐ CLAVE: Si hay obstáculo y está bloqueando, usar su profundidad
            if self.obstacle_detected and self.use_obstacle_depth and self.obstacle_min_depth is not None:
                # Solo si el obstáculo está significativamente más cerca
                if self.obstacle_min_depth < (Z * 0.8):  # Obstáculo al menos 20% más cerca
                    Z_original = Z
                    Z = self.obstacle_min_depth
                    # Recalcular X con la nueva profundidad
                    X = (cx - self.cx) * Z / self.fx
                    
                    self.get_logger().warn(
                        f'🚧 Ajustando profundidad: {Z_original:.2f}m → {Z:.2f}m (obstáculo)',
                        throttle_duration_sec=1.0
                    )
            
            pose_msg = PoseStamped()
            pose_msg.header = header
            pose_msg.pose.position.x = float(X)
            pose_msg.pose.position.y = float(Y)
            pose_msg.pose.position.z = float(Z)
            pose_msg.pose.orientation.w = 1.0
            
            self.pose_pub.publish(pose_msg)
            
            if self.obstacle_detected:
                if self.obstacle_coverage > self.large_obstacle_thresh:
                    obs_info = f" 🧱 MURO ({self.obstacle_coverage*100:.0f}%)"
                else:
                    obs_info = f" 📦 OBS ({self.obstacle_coverage*100:.0f}%)"
            else:
                obs_info = ""
            
            self.get_logger().info(
                f'Target ID {self.selected_id}: X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m{obs_info}'
            )
    
    def publish_debug_image(self):
        """Publicar imagen con todos los tracks y obstáculos"""
        if self.latest_depth is None:
            return
        
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(self.latest_depth, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        # Dibujar región del camino verificada
        if self.camera_info_received:
            img_height, img_width = self.latest_depth.shape
            img_center_x = int(self.cx)
            path_width_pixels = int((self.path_width * self.fx) / 1.0)
            
            path_x_start = int(img_center_x - path_width_pixels / 2)
            path_x_end = int(img_center_x + path_width_pixels / 2)
            path_y_start = int(img_height * 0.5)
            path_y_end = img_height
            
            # Color según si hay obstáculo
            path_color = (0, 0, 255) if self.obstacle_detected else (0, 255, 0)
            cv2.rectangle(depth_colormap, 
                         (path_x_start, path_y_start), 
                         (path_x_end, path_y_end), 
                         path_color, 2)
            
            # Texto con ancho del camino
            path_text = f'Path: {self.path_width:.2f}m'
            cv2.putText(depth_colormap, path_text, (path_x_start, path_y_start - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, path_color, 2)
            
            # Si hay obstáculo, marcar su posición
            if self.obstacle_detected and self.obstacle_min_depth is not None:
                # Calcular píxel X del obstáculo
                obs_px_x = int(self.cx + (self.obstacle_lateral_pos * self.fx / self.obstacle_min_depth))
                obs_px_y = int(img_height * 0.75)  # Mitad del path_region
                cv2.circle(depth_colormap, (obs_px_x, obs_px_y), 10, (0, 0, 255), -1)
                cv2.putText(depth_colormap, f'{self.obstacle_min_depth:.2f}m', 
                           (obs_px_x + 15, obs_px_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Dibujar tracks
        for track_id, track in self.tracks.items():
            x1, y1, x2, y2 = [int(v) for v in track['bbox']]
            cx, cy = [int(v) for v in track['center']]
            
            if track_id == self.selected_id:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (255, 0, 0)
                thickness = 2
            
            cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(depth_colormap, (cx, cy), 5, color, -1)
            
            text = f'ID:{track_id}'
            if track_id == self.selected_id:
                text += ' [TARGET]'
            
            cv2.putText(depth_colormap, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Info text
        info_text = f'Tracks: {len(self.tracks)} | Target: {self.selected_id if self.selected_id else "None"}'
        if self.obstacle_detected:
            if self.obstacle_coverage > self.large_obstacle_thresh:
                info_text += f' | MURO: {self.obstacle_coverage*100:.0f}%'
            else:
                info_text += f' | OBS: {self.obstacle_coverage*100:.0f}%'
        
        cv2.putText(depth_colormap, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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