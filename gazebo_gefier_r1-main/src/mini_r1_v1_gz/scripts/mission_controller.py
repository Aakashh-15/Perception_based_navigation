#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')
        
        # Sub/Pub
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(String, '/vision_event', self.event_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # STATE MACHINE 
        self.state = "FOLLOW_RIGHT" 
        self.state_start_time = 0.0

        # --- MIT RSS Wall Following Parameters (Updated to your specs) ---
        self.desired_distance = 0.36    # meters from the wall
        self.lookahead = 0.35           # (L) meters to project forward 
        self.kp = 3.0                   # Proportional gain (Steering intensity)
        self.kd = 0.15                  # Derivative gain (Dampens oscillations)
        
        self.base_speed = 0.25          # Fast straight speed
        self.cornering_speed = 0.05     # Slower speed for sharp turns

        self.prev_error = 0.0
        self.last_time = self.get_clock().now()

        self.get_logger().info("🏎️ Controller Online: Following RIGHT wall.")

    def event_callback(self, msg):
        event = msg.data
        now = self.get_clock().now().nanoseconds / 1e9

        if event == "ARUCO_1" and self.state == "FOLLOW_RIGHT":
            self.state = "WAIT_U_TURN"
            self.state_start_time = now
            self.get_logger().warn("ID 1 Received! Pausing 1 second before U-Turn...")
            
        elif event == "ARUCO_2" and self.state == "FOLLOW_RIGHT":
            self.state = "WAIT_LEFT_FOLLOW"
            self.state_start_time = now
            self.get_logger().warn("ID 2 Received! Pausing 1.5 sec before Left Wall Follow...")
            
        elif event == "RED_TILE" and self.state != "STOPPED" and self.state != "WAIT_STOP":
            self.state = "WAIT_STOP"
            self.state_start_time = now
            self.get_logger().error("Red Tile! Driving to center for 1.5s then stopping.")

    def get_range(self, ranges, angle, angle_min, angle_increment):
        index = int((angle - angle_min) / angle_increment)
        if index < 0 or index >= len(ranges): return 2.0 
        dist = ranges[index]
        if math.isinf(dist) or math.isnan(dist) or dist == 0.0: return 3.0 
        return dist

    def execute_wall_follow(self, msg, side="RIGHT"):
        # Geometry setup based on side
        sign = -1.0 if side == "RIGHT" else 1.0
        angle_b = sign * (math.pi / 2.0) 
        angle_a = sign * (math.pi / 4.0)

        b = self.get_range(msg.ranges, angle_b, msg.angle_min, msg.angle_increment)
        a = self.get_range(msg.ranges, angle_a, msg.angle_min, msg.angle_increment)

        theta = abs(angle_b - angle_a)
        alpha = math.atan2((a * math.cos(theta)) - b, a * math.sin(theta))
        D_t = b * math.cos(alpha)
        D_t1 = D_t + (self.lookahead * math.sin(alpha))

        error = D_t1 - self.desired_distance

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0: dt = 0.01 
        derivative = (error - self.prev_error) / dt
        
        # Calculate Steering (Invert if Right Wall)
        steering_angle = (self.kp * error) + (self.kd * derivative)
        if side == "RIGHT":
            steering_angle = -steering_angle
            
        # Limits matching your script
        steering_angle = max(-1.5, min(1.5, steering_angle)) 
        
        self.prev_error = error
        self.last_time = now

        t = Twist()
        # Dynamic Velocity Scaling using your parameters
        if abs(steering_angle) > 0.4:
            t.linear.x = self.cornering_speed
        else:
            t.linear.x = self.base_speed
            
        t.angular.z = steering_angle
        self.cmd_pub.publish(t)

    def scan_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        t = Twist()

        # 1. STOPPED STATE
        if self.state == "STOPPED":
            self.cmd_pub.publish(t) # Publish zero velocity
            return

        # 2. WAIT FOR U-TURN (1 second delay)
        if self.state == "WAIT_U_TURN":
            if now - self.state_start_time < 1.0:
                t.linear.x = 0.1 # Creep forward slightly while waiting
                self.cmd_pub.publish(t)
            else:
                self.state = "EXECUTE_U_TURN"
                self.state_start_time = now
                self.get_logger().info("Executing U-Turn...")
            return

        # 3. EXECUTE U-TURN
        if self.state == "EXECUTE_U_TURN":
            # Time to turn ~180 degrees (Tune '2.5' seconds and '1.25' speed for your bot)
            if now - self.state_start_time < 3.5: 
                t.angular.z = 1.25 # Spin Left
                self.cmd_pub.publish(t)
            else:
                self.state = "FOLLOW_RIGHT" # Resume right wall following
                self.get_logger().info("U-Turn Complete. Resuming RIGHT wall.")
            return

        # 4. WAIT FOR LEFT WALL FOLLOW (1.5 second delay)
        if self.state == "WAIT_LEFT_FOLLOW":
            if now - self.state_start_time < 1.5:
                t.linear.x = 0.15 # Keep driving forward into the junction
                self.cmd_pub.publish(t)
            else:
                self.state = "FOLLOW_LEFT"
                self.get_logger().info("Switching to LEFT wall following.")
            return

        # 5. WAIT TO STOP (1.5 seconds on Red Tile)
        if self.state == "WAIT_STOP":
            if now - self.state_start_time < 1.5:
                t.linear.x = 0.05 # Drive forward into the center of the red tile
                self.cmd_pub.publish(t)
            else:
                self.state = "STOPPED"
                self.get_logger().error("MISSION COMPLETE. Robot secured on Red Tile.")
            return

        # 6. NORMAL WALL FOLLOWING
        if self.state == "FOLLOW_RIGHT":
            self.execute_wall_follow(msg, side="RIGHT")
        elif self.state == "FOLLOW_LEFT":
            self.execute_wall_follow(msg, side="LEFT")

def main(args=None):
    rclpy.init(args=args)
    node = MissionController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()