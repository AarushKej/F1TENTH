#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ObstacleAvoidance(Node):
    """Obstacle avoidance with reduced oscillation"""

    def __init__(self):
        super().__init__('obstacle_avoidance')

        # Configuration parameters
        self.safety_distance = 1.0    # Minimum distance to obstacles (meters)
        self.max_range = 5.0         # Maximum LiDAR range (meters)
        self.max_speed = 1.5         # Maximum speed (m/s)
        self.min_speed = 0.3         # Minimum speed (m/s)
        self.max_steering = 0.5      # Max steering angle (radians, ~28°)
        self.fov = 2 * np.pi / 3     # FOV (±60°)
        self.steering_damping = 0.8  # Higher damping for smoother steering

        # Topics
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Subscriber and Publisher
        self.lidar_sub = self.create_subscription(
            LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)

        # State variables
        self.angles = None
        self.angles_initialized = False
        self.previous_steering = 0.0
        self.previous_speed = self.max_speed

    def preprocess_lidar(self, ranges):
        """Preprocess LiDAR data"""
        if not ranges:
            return np.array([])

        ranges = np.array(ranges)
        ranges = np.nan_to_num(ranges, nan=self.max_range, posinf=self.max_range, neginf=0.0)
        return np.clip(ranges, 0.0, self.max_range)

    def calculate_avoidance(self, ranges, angles):
        """Calculate steering and speed with reduced oscillation"""
        # Filter to FOV
        fov_mask = np.abs(angles) <= self.fov / 2
        ranges_fov = ranges[fov_mask]
        angles_fov = angles[fov_mask]

        if len(ranges_fov) == 0:
            return 0.0, self.max_speed

        # Softer weighting: reduce influence of distant obstacles
        weights = 1.0 / (ranges_fov + 0.5)  # Less aggressive than +0.1
        steering_forces = weights * np.sin(angles_fov)

        # Moderate response to close obstacles
        close_mask = ranges_fov < self.safety_distance
        if np.any(close_mask):
            close_weights = 1.0 / (ranges_fov[close_mask] + 0.5)
            close_forces = close_weights * np.sin(angles_fov[close_mask])
            steering_force = np.sum(close_forces) * 1.5  # Reduced from 2
        else:
            steering_force = np.sum(steering_forces)

        # Apply steering with stronger damping
        raw_steering = np.clip(steering_force, -self.max_steering, self.max_steering)
        steering_angle = (1 - self.steering_damping) * raw_steering + \
                         self.steering_damping * self.previous_steering
        self.previous_steering = steering_angle

        # Speed calculation with smoothing
        min_distance = np.min(ranges_fov[ranges_fov > 0]) if np.any(ranges_fov > 0) else self.max_range
        speed_factor = min(1.0, min_distance / self.safety_distance)
        target_speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
        speed = 0.7 * target_speed + 0.3 * self.previous_speed  # Smooth speed transitions
        speed = np.clip(speed, self.min_speed, self.max_speed)
        self.previous_speed = speed

        # Add hysteresis: only change steering direction if significant
        if abs(steering_angle) < 0.1 and abs(self.previous_steering) > 0.1:
            steering_angle = self.previous_steering * 0.9  # Maintain direction slightly

        return steering_angle, speed

    def lidar_callback(self, data):
        """Process LiDAR data and generate avoidance commands"""
        proc_ranges = self.preprocess_lidar(data.ranges)
        if len(proc_ranges) == 0:
            self.get_logger().warning("No valid LiDAR data received")
            return

        # Initialize angles once
        if not self.angles_initialized:
            self.angles = np.linspace(data.angle_min, data.angle_max, len(proc_ranges))
            self.angles_initialized = True

        # Calculate avoidance maneuver
        steering_angle, speed = self.calculate_avoidance(proc_ranges, self.angles)

        # Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = float(-steering_angle)  # Negative to turn away
        drive_msg.drive.speed = float(speed)
        self.drive_pub.publish(drive_msg)

        self.get_logger().info(
            f"Avoidance: Angle={np.degrees(steering_angle):.2f}°, Speed={speed:.2f} m/s"
        )

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ObstacleAvoidance()
        print("ObstacleAvoidance Node Started")
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()