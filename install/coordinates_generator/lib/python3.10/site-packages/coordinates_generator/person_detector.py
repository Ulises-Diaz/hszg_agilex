#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector')
        
        self.declare_parameter('model_path', '/home/agilex/agilex_mx/src/coordinates_generator/models/yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('detections_topic', '/person_detections')
        self.declare_parameter('debug_topic', '/person_detector/debug')
        self.declare_parameter('person_class_id', 0)  # Only Detec persons
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('img_size', 640)  # Image size

        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        rgb_topic = self.get_parameter('rgb_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        debug_topic = self.get_parameter('debug_topic').value
        self.person_class_id = self.get_parameter('person_class_id').value
        use_gpu = self.get_parameter('use_gpu').value
        self.img_size = self.get_parameter('img_size').value

        self.bridge = CvBridge()
        
        self.model = YOLO(model_path)
        
        self.rgb_sub = self.create_subscription(
            Image,
            rgb_topic,
            self.image_callback,
            10
        )
        
        self.detections_pub = self.create_publisher(Detection2DArray, detections_topic, 10)
        self.debug_pub = self.create_publisher(Image, debug_topic, 10)
            
    def image_callback(self, msg):
        try:
            # Convert image msg to opencv image to visualization
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # YOLO DETECTOR
            results = self.model(
                image, 
                conf=self.confidence_threshold, 
                verbose=False,
                imgsz=self.img_size,
                classes=[self.person_class_id]  # People Detection only
            )
            
            # detection msg
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header
            
            # Result processing
            for result in results:
                for box in result.boxes:
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    detection = Detection2D()
                    detection.header = msg.header
                    
                    detection.bbox.center.position.x = float((x1 + x2) / 2.0)
                    detection.bbox.center.position.y = float((y1 + y2) / 2.0)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)
                    
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(cls)
                    hypothesis.hypothesis.score = conf
                    detection.results.append(hypothesis)
                    
                    detections_msg.detections.append(detection)
            
            # Publish Detections
            self.detections_pub.publish(detections_msg)
            
            # Debug image to see detections
            if len(detections_msg.detections) > 0:
                self.publish_debug(image, detections_msg)
                self.get_logger().info(f'Detectadas {len(detections_msg.detections)} personas')
                
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
    
    def publish_debug(self, image, detections_msg):
        """Dibujar detecciones en la imagen"""
        debug_img = image.copy()
        
        for i, det in enumerate(detections_msg.detections):
            
            cx = int(det.bbox.center.position.x)
            cy = int(det.bbox.center.position.y)
            w = int(det.bbox.size_x)
            h = int(det.bbox.size_y)
            
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            
            # conf = det.results[0].hypothesis.score
            
            color = self.get_color(i)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(debug_img, (cx, cy), 5, color, -1)
            
            text = f'Person Detected ID:{i}'
            cv2.putText(debug_img, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Publish image to see detections
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, 'bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Error debug: {str(e)}')
    
    def get_color(self, idx):
        colors = [
            (0, 255, 0),    
        ]
        return colors[idx % len(colors)]


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()
    
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