#!/usr/bin/env python

import rospy
import numpy as np
from scipy.spatial import KDTree
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1


class WaypointUpdater(object):
    def __init__(self):
        print("Start")
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.light_wp_idx = -1
        
        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # Hz
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree: # Wait till have data
                lane = self.get_lane()
                self.final_waypoints_pub.publish(lane)
            rate.sleep()

    def pose_cb(self, pose):
        self.pose = pose

    def waypoints_cb(self, waypoints):
        if not self.base_waypoints:
            self.base_waypoints = waypoints
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.light_wp_idx = msg.data                
        

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
    
    def find_closest_waypoint_idx(self):        
        pos = self.pose.pose.position
        x, y = pos.x, pos.y
        nearest_wp_idx = self.waypoint_tree.query([x, y], 1)[1] # Returns the index of the nearest WP
        nearest_wp = self.waypoints_2d[nearest_wp_idx]
        prev_wp = self.waypoints_2d[nearest_wp_idx - 1] 

        # calculate if the nearest point is in front or behind the current position
        nearest_wp_v = np.array(nearest_wp)
        prev_wp_v = np.array(prev_wp)
        pos_v = np.array([x, y])
        behind = np.dot(nearest_wp_v - prev_wp_v, pos_v - nearest_wp_v) > 0

        if behind:
            return (nearest_wp_idx + 1) % len(self.waypoints_2d)
        else:
            return nearest_wp_idx

    def get_lane(self):
        lane = Lane()
        idx = self.find_closest_waypoint_idx()

        base_waypoints = self.base_waypoints.waypoints[idx:idx + LOOKAHEAD_WPS] 

        # Wrap around. Not really necessary as the car stops at the last point.
        if len(base_waypoints) < LOOKAHEAD_WPS:            
            base_waypoints.extend(self.base_waypoints.waypoints[:LOOKAHEAD_WPS - len(base_waypoints)])
            

        if self.light_wp_idx == -1 or (self.light_wp_idx >= ((idx + LOOKAHEAD_WPS) % len(self.waypoints_2d))):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate(base_waypoints, idx)            
        return lane

    def decelerate(self, base_waypoints, start_idx):
        lane = []
        for i, wp in enumerate(base_waypoints):
            wp2 = Waypoint()
            wp2.pose = wp.pose

            stop_idx = max(self.light_wp_idx - start_idx - 2, 0) # Allow some margin to the front of the car
            d = self.distance(base_waypoints, i, stop_idx)
            v = math.sqrt(2 * MAX_DECEL * d)
            if v < 1.:
                v = 0.
            
            self.set_waypoint_velocity(wp2, min(self.get_waypoint_velocity(wp), v))
            
            if i <= stop_idx:
                lane.append(wp2)
            
        return lane

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
