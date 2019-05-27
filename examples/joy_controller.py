#!/usr/bin/env python
PKG = 'numpy_tutorial'
import roslib; roslib.load_manifest(PKG)

import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Joy
import sys
sys.path=sys.path[3:]
import cv2
import numpy as np
import matplotlib.pyplot as plt

pub = rospy.Publisher('linear_vel_command', numpy_msg(Floats),queue_size=10)
# Author: Andrew Dai
def callback(data):
    vel_z=4*data.axes[1]/100
    vel_x=4*data.axes[0]/100
    #negative sign in vel_z because agent eyes look at negative z axis
    vel_to_publish=np.float32([-vel_z,-vel_x])
    print("joy_controller published" + str(vel_to_publish))
    pub.publish(vel_to_publish)

# Intializes everything
def start():
    rospy.Subscriber("joy", Joy, callback)
    # starts the node
    rospy.init_node('Joy2Turtle')
    rospy.spin()

if __name__ == '__main__':
    start()