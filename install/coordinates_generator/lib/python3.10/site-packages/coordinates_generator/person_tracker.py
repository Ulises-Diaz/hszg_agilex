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
from filterpy.kalman import KalmanFilter


class PersonTrackerNode(Node):
    def __init__(self):
        super().__init__('person_tracker')
        
        # Parámetros existentes
        self.declare_parameter('detections_topic', '/person_detections')
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')  # ⭐ AGREGADO
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
        
        # ⭐ PARÁMETROS KALMAN + COLOR
        self.declare_parameter('enable_kalman', True)
        self.declare_parameter('enable_color_matching', True)
        self.declare_parameter('kalman_max_distance', 2.0)  # metros - aumentado
        self.declare_parameter('color_weight', 0.6)  # peso del color en score
        self.declare_parameter('distance_weight', 0.4)  # peso de distancia en score
        self.declare_parameter('min_match_score', 0.4)  # umbral mínimo - bajado
        self.declare_parameter('color_update_rate', 0.15)  # tasa de actualización del histograma
        self.declare_parameter('target_max_age', 90)  # ⭐ NUEVO: frames extra para target (3x normal)
        self.declare_parameter('occlusion_mode_threshold', 5)  # ⭐ frames sin ver para entrar en modo oclusión
        self.declare_parameter('position_jump_penalty', 0.5)  # ⭐ penalización por salto grande de posición
        
        # PARÁMETROS DE GEOMETRÍA DEL ROBOT
        self.declare_parameter('robot_width', 0.22)
        self.declare_parameter('robot_length', 0.45)
        
        # Parámetros de detección de obstáculos
        self.declare_parameter('enable_obstacle_detection', True)
        self.declare_parameter('obstacle_min_distance', 0.5)
        self.declare_parameter('obstacle_width_threshold', 0.3)
        self.declare_parameter('path_check_width_multiplier', 1.8)
        self.declare_parameter('large_obstacle_threshold', 0.6)
        self.declare_parameter('obstacle_depth_override', True)
        
        # Obtener parámetros
        detections_topic = self.get_parameter('detections_topic').value
        rgb_topic = self.get_parameter('rgb_topic').value  # ⭐
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
        
        self.enable_kalman = self.get_parameter('enable_kalman').value
        self.enable_color = self.get_parameter('enable_color_matching').value
        self.kalman_max_dist = self.get_parameter('kalman_max_distance').value
        self.color_weight = self.get_parameter('color_weight').value
        self.distance_weight = self.get_parameter('distance_weight').value
        self.min_match_score = self.get_parameter('min_match_score').value
        self.color_update_rate = self.get_parameter('color_update_rate').value
        self.target_max_age = self.get_parameter('target_max_age').value  # ⭐
        self.occlusion_threshold = self.get_parameter('occlusion_mode_threshold').value  # ⭐
        self.position_jump_penalty = self.get_parameter('position_jump_penalty').value  # ⭐
        
        self.enable_obstacle = self.get_parameter('enable_obstacle_detection').value
        self.obstacle_min_dist = self.get_parameter('obstacle_min_distance').value
        self.obstacle_width_thresh = self.get_parameter('obstacle_width_threshold').value
        self.large_obstacle_thresh = self.get_parameter('large_obstacle_threshold').value
        self.use_obstacle_depth = self.get_parameter('obstacle_depth_override').value
        
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
        self.latest_rgb = None  # ⭐
        
        # ⭐ Filtro de Kalman y color para persona seleccionada
        self.kalman_filter = None
        self.target_histogram = None
        self.target_last_valid_pos = None  # ⭐ NUEVO: última posición 3D válida del target
        self.target_last_depth = None  # ⭐ NUEVO: última profundidad válida
        
        # Obstacle detection
        self.obstacle_detected = False
        self.obstacle_side = "none"
        self.obstacle_coverage = 0.0
        self.obstacle_min_depth = None
        self.obstacle_lateral_pos = 0.0
        
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
        
        # ⭐ Suscriptor RGB
        self.rgb_sub = self.create_subscription(
            Image,
            rgb_topic,
            self.rgb_callback,
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
        
        self.get_logger().info('🤖 Person Tracker con KALMAN + COLOR HÍBRIDO')
        self.get_logger().info(f'✅ Kalman: {self.enable_kalman} | Color: {self.enable_color}')
        self.get_logger().info(f'📊 Pesos - Color: {self.color_weight:.1f} | Distancia: {self.distance_weight:.1f}')
        self.get_logger().info(f'🎯 Umbral match: {self.min_match_score:.2f}')
        self.get_logger().info(f'Auto-select closest: {self.auto_select}')
        self.get_logger().info(f'📐 Ancho robot: {robot_width}m')
        self.get_logger().info(f'🛣️  Ancho camino verificado: {self.path_width:.2f}m')
    
    # ⭐ Callback RGB
    def rgb_callback(self, msg):
        """Almacenar imagen RGB"""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Error RGB: {str(e)}')
    
    # ⭐ Inicializar Kalman
    def init_kalman_filter(self, x, y):
        """Inicializar Kalman para la persona seleccionada"""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([x, y, 0., 0.])  # [x, y, vx, vy]
        
        dt = 0.1  # ~10 Hz
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        
        kf.R *= 5    # ruido de medición
        kf.P *= 10   # incertidumbre inicial
        kf.Q *= 0.1  # ruido del proceso
        
        return kf
    
    # ⭐ Extraer histograma de color
    def extract_histogram(self, bbox):
        """Extrae histograma HSV del torso de la persona"""
        if self.latest_rgb is None:
            return None
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Validar bbox
        h, w = self.latest_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extraer región del torso (60% superior)
        bbox_height = y2 - y1
        torso_y_end = int(y1 + bbox_height * 0.6)
        torso_region = self.latest_rgb[y1:torso_y_end, x1:x2]
        
        if torso_region.size == 0:
            return None
        
        # Convertir a HSV (más robusto a iluminación)
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        
        # Histograma en H y S (ignoramos V por cambios de iluminación)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        
        return hist.flatten()
    
    # ⭐ Comparar histogramas
    def compare_histograms(self, hist1, hist2):
        """Retorna similitud entre 0 y 1"""
        if hist1 is None or hist2 is None:
            return 0.0
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
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
            
            # ⭐ Inicializar Kalman y capturar histograma
            track = self.tracks[track_id]
            cx, cy = track['center']
            depth = self.get_depth_at_point(int(cx), int(cy))
            
            if depth is not None and self.camera_info_received:
                # Convertir a coordenadas 3D
                X = (cx - self.cx) * depth / self.fx
                Y = (cy - self.cy) * depth / self.fy
                
                if self.enable_kalman:
                    self.kalman_filter = self.init_kalman_filter(X, Y)
                    self.get_logger().info(f'✓ Kalman inicializado en ({X:.2f}, {Y:.2f})')
                
                if self.enable_color:
                    self.target_histogram = self.extract_histogram(track['bbox'])
                    if self.target_histogram is not None:
                        self.get_logger().info('✓ Histograma de color capturado')
                    else:
                        self.get_logger().warn('⚠️  No se pudo capturar histograma')
            
            self.get_logger().info(f'✓ Persona {track_id} seleccionada')
            status_msg = String()
            status_msg.data = f'tracking_id_{track_id}'
            self.status_pub.publish(status_msg)
            
        elif track_id == -1:
            self.selected_id = None
            self.kalman_filter = None
            self.target_histogram = None
            self.target_last_valid_pos = None  # ⭐
            self.target_last_depth = None  # ⭐
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
            
            # ⭐ LÓGICA HÍBRIDA: Kalman + Color para target
            if self.selected_id is not None and (self.enable_kalman or self.enable_color):
                self.update_tracks_hybrid(current_detections, msg.header)
            else:
                # Lógica original con IoU
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
    
    # ⭐ NUEVA FUNCIÓN HÍBRIDA: Kalman + Color
    def update_tracks_hybrid(self, detections, header):
        """Actualizar tracks usando Kalman + Color para target, IoU para resto"""
        
        # Predicción de Kalman
        predicted_pos = None
        predicted_cx, predicted_cy = None, None
        cx_prev, cy_prev = None, None
        
        if self.kalman_filter is not None and self.selected_id in self.tracks:
            self.kalman_filter.predict()
            predicted_pos = self.kalman_filter.x[:2]  # [X, Y] en metros
            
            # Convertir predicción 3D a píxeles para búsqueda
            track = self.tracks[self.selected_id]
            cx_prev, cy_prev = track['center']
            depth_prev = self.get_depth_at_point(int(cx_prev), int(cy_prev))
            
            if depth_prev is not None and depth_prev > 0:
                predicted_cx = self.cx + (predicted_pos[0] * self.fx / depth_prev)
                predicted_cy = self.cy + (predicted_pos[1] * self.fy / depth_prev)
        elif self.selected_id in self.tracks:
            # Si no hay Kalman pero sí target, obtener posición anterior
            track = self.tracks[self.selected_id]
            cx_prev, cy_prev = track['center']
        
        # Marcar todos como no actualizados
        for track in self.tracks.values():
            track['updated'] = False
        
        matched_det_indices = set()
        
        # PASO 1: Match híbrido para el TARGET
        if self.selected_id in self.tracks and len(detections) > 0:
            best_match = None
            best_score = -1
            best_det_idx = -1
            best_dist_3d = float('inf')
            best_color_sim = 0.0
            
            # ⭐ Verificar si estamos en modo oclusión
            current_age = self.tracks[self.selected_id]['age']
            in_occlusion_mode = current_age >= self.occlusion_threshold
            
            if in_occlusion_mode:
                self.get_logger().warn(
                    f'🔍 MODO OCLUSIÓN (age={current_age}) - Búsqueda MUY restrictiva',
                    throttle_duration_sec=0.5
                )
            
            for idx, det in enumerate(detections):
                cx, cy = det['center']
                depth = self.get_depth_at_point(int(cx), int(cy))
                
                if depth is None or depth <= 0:
                    continue
                
                # Convertir a 3D
                X = (cx - self.cx) * depth / self.fx
                Y = (cy - self.cy) * depth / self.fy
                
                # Score de distancia (Kalman)
                dist_score = 0.0
                dist_3d = float('inf')
                
                if predicted_pos is not None:
                    dist_3d = np.sqrt((X - predicted_pos[0])**2 + (Y - predicted_pos[1])**2)
                    
                    # ⭐ En modo oclusión, ser MÁS restrictivo con la distancia
                    max_dist_allowed = self.kalman_max_dist
                    if in_occlusion_mode:
                        max_dist_allowed = self.kalman_max_dist * 0.6  # Solo 60% del máximo
                    
                    if dist_3d > max_dist_allowed:
                        continue
                    
                    # Normalizar distancia a [0, 1]
                    dist_score = 1.0 - (dist_3d / self.kalman_max_dist)
                    
                    # ⭐ NUEVO: Penalizar saltos grandes de posición si hay última posición válida
                    if self.target_last_valid_pos is not None:
                        last_X, last_Y = self.target_last_valid_pos
                        position_jump = np.sqrt((X - last_X)**2 + (Y - last_Y)**2)
                        
                        # Si el salto es > 1.5m, aplicar penalización
                        if position_jump > 1.5:
                            jump_penalty = min(position_jump / 3.0, 1.0) * self.position_jump_penalty
                            dist_score = dist_score * (1.0 - jump_penalty)
                            self.get_logger().warn(
                                f'⚠️  Penalización por salto: {position_jump:.2f}m (score reducido)',
                                throttle_duration_sec=1.0
                            )
                
                elif cx_prev is not None and cy_prev is not None:
                    # Sin Kalman, usar distancia 2D
                    dist_2d = np.sqrt((cx - cx_prev)**2 + (cy - cy_prev)**2)
                    max_pixel_dist = 200
                    if dist_2d > max_pixel_dist:
                        continue
                    dist_score = 1.0 - (dist_2d / max_pixel_dist)
                    dist_3d = dist_2d / 100.0
                else:
                    dist_score = 0.5
                
                # Score de color
                color_sim = 0.0
                if self.enable_color and self.target_histogram is not None:
                    det_histogram = self.extract_histogram(det['bbox'])
                    color_sim = self.compare_histograms(self.target_histogram, det_histogram)
                    
                    # ⭐ En modo oclusión, EXIGIR color similar
                    if in_occlusion_mode and color_sim < 0.5:
                        self.get_logger().info(
                            f'❌ Rechazado: color muy diferente ({color_sim:.2f}) en modo oclusión',
                            throttle_duration_sec=1.0
                        )
                        continue
                else:
                    color_sim = 0.5
                
                # Score combinado (ponderado)
                combined_score = (color_sim * self.color_weight + 
                                dist_score * self.distance_weight)
                
                # ⭐ En modo oclusión, aumentar umbral mínimo
                effective_min_score = self.min_match_score
                if in_occlusion_mode:
                    effective_min_score = max(self.min_match_score, 0.6)  # Mínimo 0.6
                
                if combined_score > best_score and combined_score >= effective_min_score:
                    best_score = combined_score
                    best_match = det
                    best_det_idx = idx
                    best_dist_3d = dist_3d
                    best_color_sim = color_sim
            
            # Si encontramos match válido
            if best_match is not None:
                track = self.tracks[self.selected_id]
                track['bbox'] = best_match['bbox']
                track['center'] = best_match['center']
                track['confidence'] = best_match['confidence']
                track['age'] = 0
                track['updated'] = True
                track['header'] = header
                matched_det_indices.add(best_det_idx)
                
                # Actualizar Kalman
                if self.kalman_filter is not None:
                    cx, cy = best_match['center']
                    depth = self.get_depth_at_point(int(cx), int(cy))
                    if depth is not None and depth > 0:
                        X = (cx - self.cx) * depth / self.fx
                        Y = (cy - self.cy) * depth / self.fy
                        self.kalman_filter.update([X, Y])
                        
                        # ⭐ Guardar última posición válida
                        self.target_last_valid_pos = (X, Y)
                        self.target_last_depth = depth
                
                # Actualizar histograma gradualmente
                if self.enable_color and self.target_histogram is not None:
                    new_hist = self.extract_histogram(best_match['bbox'])
                    if new_hist is not None:
                        alpha = 1.0 - self.color_update_rate
                        self.target_histogram = alpha * self.target_histogram + self.color_update_rate * new_hist
                
                occlusion_tag = " [RECUPERADO]" if in_occlusion_mode else ""
                self.get_logger().info(
                    f'🎯 Match{occlusion_tag} (score:{best_score:.2f} = color:{best_color_sim:.2f} + dist:{best_dist_3d:.2f}m)',
                    throttle_duration_sec=0.5
                )
            else:
                # No hay buen match
                self.tracks[self.selected_id]['age'] += 1
                
                if in_occlusion_mode:
                    self.get_logger().warn(
                        f'🔍 Oclusión continúa (age={self.tracks[self.selected_id]["age"]}) - Usando predicción',
                        throttle_duration_sec=0.5
                    )
                else:
                    self.get_logger().warn(
                        f'⚠️  Sin match (mejor score: {best_score:.2f})',
                        throttle_duration_sec=0.5
                    )
        
        # PASO 2: Match para resto de tracks con IoU
        for track_id, track in list(self.tracks.items()):
            if track_id == self.selected_id or track['updated']:
                continue
            
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
        
        # PASO 3: Crear nuevos tracks
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
        
        # PASO 4: Eliminar tracks viejos
        to_remove = []
        for track_id, track in self.tracks.items():
            # ⭐ Usar max_age diferente para el target (más tolerante)
            max_age_threshold = self.target_max_age if track_id == self.selected_id else self.max_age
            
            if track['age'] > max_age_threshold:
                to_remove.append(track_id)
                if track_id == self.selected_id:
                    self.get_logger().warn(f'❌ Target perdido definitivamente (age > {max_age_threshold})')
                    self.selected_id = None
                    self.kalman_filter = None
                    self.target_histogram = None
                    self.target_last_valid_pos = None
                    self.target_last_depth = None
        
        for track_id in to_remove:
            del self.tracks[track_id]
            self.get_logger().info(f'Track {track_id} eliminado')
    
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
                
                # Guardar profundidad mínima del obstáculo
                self.obstacle_min_depth = float(np.min(close_obstacles))
                
                # Calcular posición lateral del obstáculo
                obstacle_mask = (path_region > 0) & (path_region < self.obstacle_min_dist * 1000)
                if np.any(obstacle_mask):
                    y_coords, x_coords = np.where(obstacle_mask)
                    obstacle_center_x_px = int(np.mean(x_coords))
                    obstacle_img_x = path_x_start + obstacle_center_x_px
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
                
                # Publicar tamaño del obstáculo
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
                    f'Lado: {self.obstacle_side} | Dist: {self.obstacle_min_depth:.2f}m',
                    throttle_duration_sec=1.0
                )
            else:
                self.obstacle_detected = False
                self.obstacle_side = "none"
                self.obstacle_coverage = 0.0
                self.obstacle_min_depth = None
                self.obstacle_lateral_pos = 0.0
                
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
            
            # Si hay obstáculo y está bloqueando, usar su profundidad
            if self.obstacle_detected and self.use_obstacle_depth and self.obstacle_min_depth is not None:
                if self.obstacle_min_depth < (Z * 0.8):
                    Z_original = Z
                    Z = self.obstacle_min_depth
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
            
            path_color = (0, 0, 255) if self.obstacle_detected else (0, 255, 0)
            cv2.rectangle(depth_colormap, 
                         (path_x_start, path_y_start), 
                         (path_x_end, path_y_end), 
                         path_color, 2)
            
            path_text = f'Path: {self.path_width:.2f}m'
            cv2.putText(depth_colormap, path_text, (path_x_start, path_y_start - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, path_color, 2)
            
            # Si hay obstáculo, marcar su posición
            if self.obstacle_detected and self.obstacle_min_depth is not None:
                obs_px_x = int(self.cx + (self.obstacle_lateral_pos * self.fx / self.obstacle_min_depth))
                obs_px_y = int(img_height * 0.75)
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
                mode_text = ""
                if self.kalman_filter:
                    mode_text += "K"
                if self.target_histogram is not None:
                    mode_text += "C"
                text += f' [{mode_text}]' if mode_text else ' [TARGET]'
            
            cv2.putText(depth_colormap, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Info text
        info_text = f'Tracks: {len(self.tracks)} | Target: {self.selected_id if self.selected_id else "None"}'
        if self.kalman_filter or self.target_histogram:
            modes = []
            if self.kalman_filter:
                modes.append("Kalman")
            if self.target_histogram is not None:
                modes.append("Color")
            info_text += f' [{"+".join(modes)}]'
        
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