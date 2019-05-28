#!/usr/bin/env python  
PKG = "numpy_tutorial"
import roslib

roslib.load_manifest(PKG)

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import tf


def handle_agent_pose(msg,agentname):
    pose_array=msg.data
    br = tf.TransformBroadcaster()
    br.sendTransform((pose_array[0], pose_array[1], 0),
                     pose_array[3:],
                     rospy.Time.now(),
                     agentname,
                     "world")
    print('handle_agent_pose_called')

if __name__ == '__main__':
    rospy.init_node('agent_tf_broadcaster')
    agentname = "bruce_agent"
    rospy.Subscriber('agent_pose',
                     numpy_msg(Floats),
                     handle_agent_pose,
                     agentname)
    rospy.spin()