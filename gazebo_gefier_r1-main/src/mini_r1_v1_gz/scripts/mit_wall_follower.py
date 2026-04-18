#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import numpy as np

class MITWallFollower(Node):
    def __init__(self):
        super().__init__('mit_wall_follower')
        
        # ROS 2 Pubs/Subs
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- MIT RSS Wall Following Parameters ---
        # 1. Lookahead & Geometry
        self.desired_distance = 0.5    # meters from the left wall
        self.lookahead_distance = 0.35 # (L) meters to project forward 
        
        # 2. PD Controller Tuning
        self.kp = 3.0   # Proportional gain (Steering intensity)
        self.kd = 0.15  # Derivative gain (Dampens oscillations)
        
        # 3. Base Speeds
        self.base_speed = 0.25         # Fast straight speed
        self.cornering_speed = 0.12    # Slower speed for sharp turns

        # State Variables
        self.prev_error = 0.0
        self.last_time = self.get_clock().now()
        
        self.get_logger().info("MIT Geometric Wall Follower Active [Left Wall]")

    def get_range(self, range_data, angle, angle_min, angle_increment):
        """Safely extracts the distance at a specific angle from the LiDAR array."""
        # Convert requested angle to array index
        index = int((angle - angle_min) / angle_increment)
        
        # Ensure index is within array bounds
        if index < 0 or index >= len(range_data):
            return 2.0 # Default safe distance if out of bounds
            
        dist = range_data[index]
        
        # Handle Inf and NaN readings from simulation/real sensors
        if math.isinf(dist) or math.isnan(dist) or dist == 0.0:
            return 3.0 # Max range assumption
            
        return dist

    def scan_callback(self, msg):
        # 1. Define the two rays (a and b)
        # Ray B: 90 degrees to the left (perpendicular to robot body)
        angle_b = math.pi / 2.0 
        
        # Ray A: 45 degrees to the left (forward-diagonal)
        angle_a = math.pi / 4.0 
        
        # Theta: The angle between Ray A and Ray B
        theta = angle_b - angle_a 

        # Extract actual distances from the LiDAR array
        b = self.get_range(msg.ranges, angle_b, msg.angle_min, msg.angle_increment)
        a = self.get_range(msg.ranges, angle_a, msg.angle_min, msg.angle_increment)

        # 2. MIT Math: Calculate Alpha (Angle of the robot relative to the wall)
        numerator = (a * math.cos(theta)) - b
        denominator = a * math.sin(theta)
        alpha = math.atan2(numerator, denominator)

        # 3. MIT Math: Calculate D_t (Current perpendicular distance to wall)
        D_t = b * math.cos(alpha)

        # 4. MIT Math: Calculate D_t+1 (Predicted distance to wall L meters ahead)
        D_t1 = D_t + (self.lookahead_distance * math.sin(alpha))

        # 5. PD Controller Error Calculation
        # If D_t1 > desired (too far), error is positive -> turn left.
        # If D_t1 < desired (too close), error is negative -> turn right.
        error = D_t1 - self.desired_distance

        # Time delta for Derivative
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            dt = 0.01 # Prevent division by zero on first tick
            
        derivative = (error - self.prev_error) / dt
        
        # Steering Output
        steering_angle = (self.kp * error) + (self.kd * derivative)
        
        # Limit max steering to prevent spinning out
        steering_angle = max(-1.5, min(1.5, steering_angle))
        
        # Save state for next tick
        self.prev_error = error
        self.last_time = now

        # 6. Dynamic Velocity Scaling
        # Slow down if the steering angle is very sharp (cornering)
        t = Twist()
        if abs(steering_angle) > 0.4:
            t.linear.x = self.cornering_speed
        else:
            t.linear.x = self.base_speed
            
        t.angular.z = steering_angle
        
        self.cmd_pub.publish(t)

def main(args=None):
    rclpy.init(args=args)
    node = MITWallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()