#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time

class SmoothAvoidance90Deg(Node):
    def __init__(self):
        super().__init__('yolo_tracker')
        
        # サブスクライバー
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # パブリッシャー
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        self.current_yaw = 0.0
        
        # 状態管理（ステート）
        self.state = "FORWARD"  # FORWARD, BRAKE, TURNING
        self.target_yaw = 0.0
        self.brake_start_time = 0.0
        
        self.get_logger().info('【体勢安定化・衝突前90度回避ノード】が起動しました。')

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            return

        img_h, img_w, _ = cv_image.shape
        
        # 前方警戒エリア（安全距離での検知）
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        roi_top = int(img_h * 0.3)
        roi_bottom = int(img_h * 0.7)
        roi_left = int(img_w * 0.25)
        roi_right = int(img_w * 0.95)
        
        roi = edges[roi_top:roi_bottom, roi_left:roi_right]
        obstacle_pixel_count = cv2.countNonZero(roi)
        
        twist = Twist()

        # 【ステート1: ブレーキ（姿勢安定化中）】
        if self.state == "BRAKE":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            cv2.putText(cv_image, 'STABILIZING POSITION...', (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 0.2秒間完全停止して揺れを抑えてから回転へ移行
            if time.time() - self.brake_start_time > 0.2:
                self.state = "TURNING"

        # 【ステート2: スムーズ90度旋回中】
        elif self.state == "TURNING":
            angle_diff = self.target_yaw - self.current_yaw
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            
            if abs(angle_diff) > 0.08:
                twist.linear.x = 0.0
                # 旋回速度を 0.45 に抑えて横揺れを抑圧
                twist.angular.z = 0.45  
                cv2.putText(cv_image, f'SMOOTH TURNING... ({int(math.degrees(abs(angle_diff)))} deg left)', 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                self.state = "FORWARD"
                self.get_logger().info('姿勢を崩さず90度旋回完了。前進を開始します。')

        # 【ステート3: 通常前進中】
        elif self.state == "FORWARD":
            if obstacle_pixel_count > 1200:
                # 障害物を前方に検知したらまず一時停止（ブレーキ）して姿勢を水平に戻す
                self.state = "BRAKE"
                self.brake_start_time = time.time()
                
                # 目標角度（+90度）の事前計算
                self.target_yaw = self.current_yaw + (math.pi / 2.0)
                self.target_yaw = math.atan2(math.sin(self.target_yaw), math.cos(self.target_yaw))
                
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                cv2.putText(cv_image, 'OBSTACLE DETECTED! PREPARING TURN...', (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                twist.linear.x = 0.15
                twist.angular.z = 0.0
                cv2.putText(cv_image, 'CLEAR: MOVING FORWARD', (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 旋回中以外は警戒エリアの枠を描画
        if self.state != "TURNING":
            cv2.rectangle(cv_image, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)

        self.cmd_pub.publish(twist)
        cv2.imshow("Camera Avoidance 90Deg", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SmoothAvoidance90Deg()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
