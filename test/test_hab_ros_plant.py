import pytest
import sys

sys.path.append('/home/bruce/catkin_ws/devel/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from geometry_msgs.msg import Twist
import threading

sys.path = [
    b for b in sys.path if "2.7" not in b
]  # remove path's related to ROS from environment or else certain packages like cv2 can't be imported

import habitat
import habitat_ros


def test_env_init():

    my_env = habitat_ros.sim_env(env_config_file="configs/tasks/pointnav_rgbd.yaml")

    assert my_env.observations != None

def test_env_update_linear():
    my_env = habitat_ros.sim_env(env_config_file="configs/tasks/pointnav_rgbd.yaml")
    initial_observations = my_env.observations
    #my_env.start()
    my_env.set_linear_velocity(1,2)

    my_env.update_orientation()
    post_observations= my_env.observations

    import numpy as np
    assert np.array_equal(initial_observations['depth'],post_observations['depth'])==False
    

def test_env_update_angular():
    my_env = habitat_ros.sim_env(env_config_file="configs/tasks/pointnav_rgbd.yaml")
    initial_observations = my_env.observations
    #my_env.start()
    my_env.set_yaw(1)
    my_env.update_orientation()
    post_observations= my_env.observations

    import numpy as np
    assert np.array_equal(initial_observations['depth'],post_observations['depth'])==False
    