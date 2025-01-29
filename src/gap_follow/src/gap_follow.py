#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import math
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    """ 
    Implement Reactive Follow Gap on the car
    """
    def __init__(self):
        super().__init__('gap_follow')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Subscribe to LIDAR
        self.lidar_sub = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.lidar_callback,
            10
        )

        # Publish to drive
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array:
            1. Smooth each value over a sliding window.
            2. Reject high values (> 3m) and low values (< 0.8m).
        """
        ranges = np.array(ranges)

        # Smooth each value using a sliding window (moving average)
        window_size = 5  # Define the window size
        smoothed_ranges = np.convolve(ranges, np.ones(window_size)/window_size, mode='same')
        # Reject high and low values by clamping
        processed_ranges = np.clip(smoothed_ranges, 0.8, 3.0)
        return processed_ranges

    def find_max_gap(self, free_space_ranges, threshold):
        """ Find the largest gap in free space ranges. """
        max_gap_len = 0
        max_gap_index = 0
        max_gap_list = []
        end_index = 0
        while max_gap_index < len(free_space_ranges):
            if free_space_ranges[max_gap_index] > threshold:
                max_gap_list.append(free_space_ranges[max_gap_index])
            else:
                if len(max_gap_list) > 0:
                    if len(max_gap_list) > max_gap_len:
                        end_index = max_gap_index - 1
                        max_gap_len = len(max_gap_list)
                    max_gap_list = []
            max_gap_index += 1
        if end_index == 0:
            threshold -= 0.3
            return self.find_max_gap(free_space_ranges,threshold)
        return end_index - max_gap_len + 1,end_index

    def find_best_point(self, start_i, end_i, ranges):
        """ Find the furthest point in the largest gap. """
        best_point = start_i + ranges[start_i:end_i+1].index(max(ranges[start_i:end_i+1]))
        return best_point

    def lidar_callback(self, data):
        """ Process LiDAR data and publish drive commands. """
        ranges = list(data.ranges)
        start_index = int(math.radians(90)/data.angle_increment)
        end_index = int((math.radians(270)/data.angle_increment))+1
        ranges = ranges[start_index:end_index]
        proc_ranges = self.preprocess_lidar(ranges)

        #Find closest point to LiDAR
        closest_point = ranges.index(min(ranges))

        #Eliminate all points inside 'bubble' (set them to zero) 
        bubble_radius = 10
        free_space_ranges = proc_ranges
        for i in range(closest_point-bubble_radius,closest_point+bubble_radius):
            free_space_ranges[i] = 0
        #free_space_ranges[closest_point-bubble_radius:closest_point + bubble_radius+1] = [0] * bubble_radius * 2
        #Find max length gap 
        start_index,end_index = self.find_max_gap(free_space_ranges,3)

        #Find the best point in the gap 
        best_point_index = self.find_best_point(start_index,end_index,ranges)
        print(ranges[best_point_index])
        #Publish Drive message
        angle = best_point_index * data.angle_increment
        angle -= math.pi/2
        print(angle)
        drive_msg = AckermannDriveStamped()
        sub_drive_msg = AckermannDrive()
        sub_drive_msg.steering_angle = angle
        sub_drive_msg.steering_angle_velocity = 8.0
        if 0 <= math.degrees(abs(angle)) <= 10:
            velocity = 0.4
        elif 10 < math.degrees(abs(angle)) <= 20:
            velocity = 0.3
        else:
            velocity = 0.2
        # Publish the drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive = AckermannDrive()
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)
        self.get_logger().info(f"Published: Angle={np.degrees(angle):.2f}°, Speed={velocity:.2f} m/s")


def main(args=None):
    rclpy.init(args=args)
    print("ReactiveFollowGap Node Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
