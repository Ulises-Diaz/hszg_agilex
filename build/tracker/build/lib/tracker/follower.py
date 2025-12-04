#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO


class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector')
        
        # Declarar parámetros
        self.declare_parameter('model_path', '/home/uli/hszg/ros2_ws/src/tracker/models/best.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('detections_topic', '/person_detector/detections')
        self.declare_parameter('debug_image_topic', '/person_detector/debug_image')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('person_class_id', 0)  # ID de clase para persona en COCO
        
        # Obtener parámetros
        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        debug_image_topic = self.get_parameter('debug_image_topic').value
        self.publish_debug = self.get_parameter('publish_debug_image').value
        self.person_class_id = self.get_parameter('person_class_id').value
        
        # Bridge de OpenCV
        self.bridge = CvBridge()
        
        # Cargar modelo YOLO
        self.get_logger().info(f'Cargando modelo YOLO desde: {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info('Modelo YOLO cargado exitosamente')
        
        # Parámetros intrínsecos de la cámara
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_received = False
        
        # Almacenar última imagen de profundidad
        self.latest_depth_image = None
        self.depth_stamp = None
        
        # Suscriptor de CameraInfo
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # Suscriptor de imagen RGB
        self.rgb_sub = self.create_subscription(
            Image,
            rgb_topic,
            self.rgb_callback,
            10
        )
        
        # Suscriptor de imagen de profundidad
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10
        )
        
        # Publisher de detecciones
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            detections_topic,
            10
        )
        
        # Publisher de imagen de debug
        if self.publish_debug:
            self.debug_image_pub = self.create_publisher(
                Image,
                debug_image_topic,
                10
            )
        
        self.get_logger().info('Nodo PersonDetector iniciado correctamente')
    
    def camera_info_callback(self, msg):
        """Callback para obtener los parámetros intrínsecos de la cámara"""
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_info_received = True
            self.get_logger().info(
                f'Parámetros de cámara recibidos: fx={self.fx:.2f}, fy={self.fy:.2f}, '
                f'cx={self.cx:.2f}, cy={self.cy:.2f}'
            )
    
    def depth_callback(self, msg):
        """Callback para almacenar la imagen de profundidad"""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Error al procesar imagen de profundidad: {str(e)}')
    
    def get_depth_at_point(self, x, y, depth_image):
        """Obtener la profundidad en un punto con región para mayor estabilidad"""
        if depth_image is None:
            return None
        
        # Región alrededor del punto
        region_size = 10
        y_min = max(0, y - region_size)
        y_max = min(depth_image.shape[0], y + region_size)
        x_min = max(0, x - region_size)
        x_max = min(depth_image.shape[1], x + region_size)
        
        depth_region = depth_image[y_min:y_max, x_min:x_max]
        
        # Filtrar valores inválidos
        valid_depths = depth_region[depth_region > 0]
        
        if len(valid_depths) > 0:
            depth = np.median(valid_depths)
            
            # Convertir a metros si está en milímetros
            if depth > 100:
                depth = depth / 1000.0
            
            return float(depth)
        
        return None
    
    def rgb_callback(self, msg):
        """Callback principal para procesar imagen RGB y detectar personas"""
        try:
            # Convertir mensaje ROS a imagen OpenCV
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detectar personas con YOLO
            results = self.model(rgb_image, conf=self.confidence_threshold, verbose=False)
            
            # Crear mensaje de detecciones
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header
            
            detected_persons = []
            
            # Procesar cada resultado
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Filtrar solo personas (ajusta person_class_id según tu modelo)
                    if cls == self.person_class_id:
                        # Obtener coordenadas del bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Crear mensaje Detection2D
                        detection = Detection2D()
                        detection.header = msg.header
                        
                        # BoundingBox2D
                        detection.bbox.center.position.x = float((x1 + x2) / 2.0)
                        detection.bbox.center.position.y = float((y1 + y2) / 2.0)
                        detection.bbox.size_x = float(x2 - x1)
                        detection.bbox.size_y = float(y2 - y1)
                        
                        # Hypothesis (clase y confianza)
                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.class_id = str(cls)
                        hypothesis.hypothesis.score = conf
                        detection.results.append(hypothesis)
                        
                        # Calcular posición 3D si tenemos profundidad y parámetros de cámara
                        if self.latest_depth_image is not None and self.camera_info_received:
                            center_x = int(detection.bbox.center.position.x)
                            center_y = int(detection.bbox.center.position.y)
                            
                            depth = self.get_depth_at_point(center_x, center_y, self.latest_depth_image)
                            
                            if depth is not None:
                                # Calcular posición 3D
                                X = (center_x - self.cx) * depth / self.fx
                                Y = (center_y - self.cy) * depth / self.fy
                                Z = depth
                                
                                # Añadir pose 3D al hypothesis
                                hypothesis.pose.pose.position.x = X
                                hypothesis.pose.pose.position.y = Y
                                hypothesis.pose.pose.position.z = Z
                                hypothesis.pose.pose.orientation.w = 1.0
                        
                        detections_msg.detections.append(detection)
                        
                        # Guardar para visualización
                        detected_persons.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'center': (int(detection.bbox.center.position.x), 
                                     int(detection.bbox.center.position.y))
                        })
            
            # Publicar detecciones
            self.detections_pub.publish(detections_msg)
            
            # Log
            if len(detected_persons) > 0:
                self.get_logger().info(f'Detectadas {len(detected_persons)} persona(s)')
            else:
                self.get_logger().debug('No se detectaron personas')
            
            # Publicar imagen de debug si está habilitado
            if self.publish_debug and len(detected_persons) > 0:
                self.publish_debug_image(rgb_image, detected_persons)
                
        except Exception as e:
            self.get_logger().error(f'Error en rgb_callback: {str(e)}')
    
    def publish_debug_image(self, image, persons):
        """Publicar imagen con visualización de todas las detecciones"""
        debug_img = image.copy()
        
        for idx, person in enumerate(persons):
            x1, y1, x2, y2 = person['bbox']
            center_x, center_y = person['center']
            conf = person['confidence']
            
            # Diferentes colores para cada persona
            color = self.get_color(idx)
            
            # Dibujar bounding box
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar punto central
            cv2.circle(debug_img, (center_x, center_y), 5, color, -1)
            
            # Añadir texto con ID y confianza
            text = f'ID:{idx} ({conf:.2f})'
            cv2.putText(debug_img, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Añadir contador
        count_text = f'Personas: {len(persons)}'
        cv2.putText(debug_img, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convertir y publicar
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Error al publicar imagen de debug: {str(e)}')
    
    def get_color(self, idx):
        """Generar color único para cada índice"""
        colors = [
            (0, 255, 0),    # Verde
            (255, 0, 0),    # Azul
            (0, 0, 255),    # Rojo
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Amarillo
            (128, 0, 128),  # Púrpura
            (255, 165, 0),  # Naranja
        ]
        return colors[idx % len(colors)]


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PersonDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {str(e)}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()