#!/usr/bin/env python
# note need to run viewer with python2!!!
PKG = "numpy_tutorial"
import roslib

roslib.load_manifest(PKG)

import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError


pub = rospy.Publisher("ros_img_depth", Image, queue_size=10)


def callback(data):
    print(rospy.get_name(), "I heard %s" % str(data.data))
    img = (np.reshape(data.data, (256, 256))).astype(np.uint8)

    image_message = CvBridge().cv2_to_imgmsg(img, encoding="mono8")
    pub.publish(image_message)


def listener():
    rospy.init_node("depth_ros_node")
    rospy.Subscriber("depth", numpy_msg(Floats), callback)
    rospy.spin()


if __name__ == "__main__":
    listener()
