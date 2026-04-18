#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os
from datetime import datetime
import time

class VisionCore(Node):
    def __init__(self):
        super().__init__('vision_core')

        self.red_detected = False
        self.red_first_seen_time = None
        
        # 1. Init YOLO
        self.model = YOLO('/home/aakash/gazebo_gefier_r1-main/gazebo_gefier_r1-main/src/mini_r1_v1_gz/scripts/best.pt').to('cpu')
        self.bridge = CvBridge()

        # 2. The Master Dictionary List (ArUco + AprilTags)
        self.all_dicts = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }

        # 3. Publisher
        self.event_pub = self.create_publisher(String, '/vision_event', 10)

        # 4. Subscriber (CRITICAL FIX: Using qos_profile_sensor_data)
        self.create_subscription(Image, '/r1_mini/camera/image_raw', self.image_callback, qos_profile_sensor_data)

        # --- STATE VARIABLES ---
        self.logo_in_view = False
        self.logo_category = "Normal" # Normal, Green, or Orange
        self.expected_aruco_id = 1    
        self.red_detected = False

        # --- CSV SETUP ---
        self.logo_csv = 'logo_log.csv'
        self.aruco_csv = 'aruco_log.csv'
        self.init_csvs()

        self.aruco_instructions = {
            1: "Take left",
            2: "Take right",
            3: "start following green",
            4: "take u turn",
            5: "start following orange"
        }

        self.get_logger().info("✅ Vision Core Online: QoS Sensor Data Active. Awaiting ID 1...")

    def init_csvs(self):
        if not os.path.exists(self.logo_csv):
            with open(self.logo_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Logo_Category'])
        if not os.path.exists(self.aruco_csv):
            with open(self.aruco_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'ArUco_ID', 'Instruction'])

    def log_to_csv(self, file_path, row_data):
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

    def image_callback(self, msg):
        if self.red_detected:
            return 

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # ==========================================
        # 1. ARUCO DETECTION (Omni-Scan with OpenCV Fallbacks)
        # ==========================================
        marker_found = False
        for dict_name, dict_id in self.all_dicts.items():
            if marker_found: break # Stop searching if we already found one this frame

            try:
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
                
                # Support both old and new OpenCV versions
                if hasattr(cv2.aruco, 'DetectorParameters_create'):
                    parameters = cv2.aruco.DetectorParameters_create()
                    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                else:
                    parameters = cv2.aruco.DetectorParameters()
                    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                    corners, ids, rejected = detector.detectMarkers(gray)

                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    for marker_id in ids.flatten():
                        m_id = int(marker_id)
                        
                        # Apply strict sequence filter
                        if m_id == self.expected_aruco_id:
                            instruction = self.aruco_instructions.get(m_id, "Unknown")
                            self.get_logger().info(f"🎯 SEQUENCE FOUND! ID: {m_id} [{dict_name}] -> {instruction}")
                            
                            self.log_to_csv(self.aruco_csv, [timestamp, m_id, instruction])
                            
                            event_msg = String()
                            event_msg.data = f"ARUCO_{m_id}"
                            self.event_pub.publish(event_msg)

                            # --- UPDATED LOGO CATEGORY LOGIC ---
                            if m_id == 3:
                                self.logo_category = "Green"
                                self.get_logger().warn("🟩 Logos -> GREEN")
                            elif m_id == 4:
                                self.logo_category = "Normal" # Between 4 and 5 it goes back to normal
                                self.get_logger().warn("⬜ Logos -> NORMAL")
                            elif m_id == 5:
                                self.logo_category = "Orange"
                                self.get_logger().warn("🟧 Logos -> ORANGE")

                            self.expected_aruco_id += 1
                            marker_found = True
                            
            except Exception:
                continue

        # ==========================================
        # 2. YOLO LOGO DETECTION
        # ==========================================
        results = self.model.predict(frame, conf=0.5, verbose=False)
        logo_found = False

        for r in results:
            for box in r.boxes:
                if r.names[int(box.cls[0])] == 'logo':
                    logo_found = True
                    break

        if logo_found and not self.logo_in_view:
            self.logo_in_view = True
            self.log_to_csv(self.logo_csv, [timestamp, self.logo_category])
            self.get_logger().info(f"📸 Logged {self.logo_category} Logo!")
        elif not logo_found:
            self.logo_in_view = False

        # ==========================================
        # 3. RED TILE DETECTION
        # ==========================================
        h, w, _ = frame.shape
        roi = frame[int(h*0.75):h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.add(mask1, mask2)

        cv2.imshow("Robot Live Camera", frame)
        cv2.waitKey(1)

        if cv2.countNonZero(red_mask) > (roi.size * 0.1):
            if self.red_first_seen_time is None:
                self.red_first_seen_time = time.time()
                self.get_logger().warn("🟡 Red tile detected, confirming for 1 second...")

        elif self.red_first_seen_time is not None and time.time() - self.red_first_seen_time >= 0.0:  # ← add None check here
            if not self.red_detected:
                self.red_detected = True
                self.event_pub.publish(String(data="RED_TILE"))
                self.get_logger().error("🛑 RED TILE SPOTTED. Signaling Controller to STOP.")
        else:
            self.red_first_seen_time = None

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VisionCore())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import csv
# import os
# from datetime import datetime

# class VisionCore(Node):
#     def __init__(self):
#         super().__init__('vision_core')
        
#         # 1. Init YOLO
#         self.model = YOLO('/home/aakash/gazebo_gefier_r1-main/gazebo_gefier_r1-main/src/mini_r1_v1_gz/scripts/best.pt').to('cpu')
#         self.bridge = CvBridge()

#         # 2. Init ArUco Dictionary (Assuming 4x4, change if you use 5x5)
#         self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
#         # self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
#         self.aruco_params = cv2.aruco.DetectorParameters()
#         self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

#         # 3. Publisher (To talk to the Controller)
#         self.event_pub = self.create_publisher(String, '/vision_event', 10)

#         # 4. Subscriber
#         self.create_subscription(Image, '/r1_mini/camera/image_raw', self.image_callback, 10)

#         # --- STATE VARIABLES ---
#         self.logo_in_view = False
#         self.logo_category = "Normal" # Changes to Green or Orange later
#         self.expected_aruco_id = 1    # Strict sequential tracker
#         self.red_detected = False

#         # --- CSV SETUP ---
#         self.logo_csv = 'logo_log.csv'
#         self.aruco_csv = 'aruco_log.csv'
#         self.init_csvs()

#         # ArUco Instructions Map
#         self.aruco_instructions = {
#             1: "Take left",
#             2: "Take right",
#             3: "start following green",
#             4: "take u turn",
#             5: "start following orange"
#         }

#         self.get_logger().info("✅ Vision Core Online: Awaiting ID 1...")

#     def init_csvs(self):
#         if not os.path.exists(self.logo_csv):
#             with open(self.logo_csv, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['Timestamp', 'Logo_Category'])
#         if not os.path.exists(self.aruco_csv):
#             with open(self.aruco_csv, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['Timestamp', 'ArUco_ID', 'Instruction'])

#     def log_to_csv(self, file_path, row_data):
#         with open(file_path, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(row_data)

#     def image_callback(self, msg):
#         if self.red_detected:
#             return # Stop processing vision if mission is over

#         # Convert ROS Image to OpenCV Image
#         frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

#         # ==========================================
#         # 1. ARUCO DETECTION (Sequential Logic)
#         # ==========================================
#         corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
        
#         if ids is not None:
#             # Draw green boxes around detected markers on the screen
#             cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
#             for marker_id in ids.flatten():
#                 if marker_id == self.expected_aruco_id:
#                     instruction = self.aruco_instructions.get(marker_id, "Unknown")
#                     self.get_logger().info(f"🎯 SEQUENCE FOUND! ID: {marker_id} -> {instruction}")
#                     self.log_to_csv(self.aruco_csv, [timestamp, marker_id, instruction])
                    
#                     event_msg = String()
#                     event_msg.data = f"ARUCO_{marker_id}"
#                     self.event_pub.publish(event_msg)

#                     if marker_id == 3:
#                         self.logo_category = "Green"
#                     elif marker_id == 5:
#                         self.logo_category = "Orange"

#                     self.expected_aruco_id += 1

#         # ==========================================
#         # 2. YOLO LOGO DETECTION
#         # ==========================================
#         results = self.model.predict(frame, conf=0.5, verbose=False)
#         logo_found = False

#         for r in results:
#             for box in r.boxes:
#                 if r.names[int(box.cls[0])] == 'logo':
#                     logo_found = True
#                     break

#         if logo_found and not self.logo_in_view:
#             self.logo_in_view = True
#             self.log_to_csv(self.logo_csv, [timestamp, self.logo_category])
#             self.get_logger().info(f"📸 Logged {self.logo_category} Logo!")
#         elif not logo_found:
#             self.logo_in_view = False

#         # ==========================================
#         # 3. RED TILE DETECTION
#         # ==========================================
#         h, w, _ = frame.shape
#         roi = frame[int(h*0.75):h, :]
#         hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
#         mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
#         mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
#         red_mask = cv2.add(mask1, mask2)
        
#         # --- DEBUG LIVE VIEW ---
#         # This opens two windows on your screen so you can see what is happening!
#         cv2.imshow("Robot Live Camera", frame)
#         cv2.imshow("Red Color Mask", red_mask)
#         cv2.waitKey(1) # Required for OpenCV windows to update

#         if cv2.countNonZero(red_mask) > (roi.size * 0.1):
#             self.red_detected = True
#             self.event_pub.publish(String(data="RED_TILE"))
#             self.get_logger().error("🛑 RED TILE SPOTTED. Signaling Controller to STOP.")

# def main(args=None):
#     rclpy.init(args=args)
#     rclpy.spin(VisionCore())
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()