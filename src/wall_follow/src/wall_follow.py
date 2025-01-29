#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow')

        # Topics
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Subscriber to LiDAR data
        self.lidar_sub = self.create_subscription(LaserScan,lidarscan_topic, self.scan_callback, 10)

        # Publisher for drive commands
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # PID gains
        self.kp = 0.5
        self.kd = 0.0
        self.ki = 0.0

        # Store PID values
        self.integral = 0.0
        self.prev_error = 0.0

        # Desired distance to the wall
        self.desired_distance = 1.0  # meters

        # Speed of the car
        self.velocity = 2.0  # m/s

    def get_range(self, range_data, angle):
        """
        Returns the range measurement at a given angle (in degrees).

        Args:
            range_data: Single range array from the LiDAR
            angle: Desired angle in degrees

        Returns:
            range: Range measurement in meters at the given angle
        """
        angle_rad = np.deg2rad(angle)
        idx = int((angle_rad - range_data.angle_min) / range_data.angle_increment)
        if 0 <= idx < len(range_data.ranges):
            distance = range_data.ranges[idx]
           
            if not np.isfinite(distance):  # Handle NaNs and infs
                return 0.0
            return distance
        return 0.0

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall on the left side (counter-clockwise).

        Args:
            range_data: Single range array from the LiDAR
            dist: Desired distance to the wall

        Returns:
            error: Calculated error
        """
        left_distance = self.get_range(range_data, 90)  # LiDAR reading at 90 degrees (left)
        error = dist - left_distance
        return error

    def pid_control(self, error, velocity):
        """
        Actuates the car using PID control.

        Args:
            error: Calculated error
            velocity: Desired velocity

        Returns:
            None
        """
        # PID calculations
        self.integral += error
        derivative = error - self.prev_error
        angle = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        # Cap the steering angle
        max_steering_angle = np.deg2rad(30)  # 30 degrees
        angle = np.clip(angle, -max_steering_angle, max_steering_angle)

        # Publish the drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = -angle
        self.drive_pub.publish(drive_msg)

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and actuate the car.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        error = self.get_error(msg, self.desired_distance)
        self.pid_control(error, self.velocity)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow = WallFollow()
    rclpy.spin(wall_follow)

    # Destroy the node explicitly
    wall_follow.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 