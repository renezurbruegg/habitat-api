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
    import numpy as np
    my_env = habitat_ros.sim_env(env_config_file="configs/tasks/pointnav_rgbd.yaml")
    my_env.set_dt(1)
    initial_observations = my_env.observations
    initial_position =  my_env.env._sim._sim.agents[0].state.position
    #my_env.start()
    my_env.set_linear_velocity(1,2)

    my_env.update_orientation()
    post_observations= my_env.observations
    post_position = my_env.env._sim._sim.agents[0].state.position

    d_should_travel = np.sqrt((1*my_env._dt)**2 + (2*my_env._dt)**2)
    assert np.array_equal(initial_observations['depth'],post_observations['depth'])==False
    assert d_should_travel*0.9<(np.linalg.norm(post_position-initial_position))<d_should_travel*1.1
    

#TODO test to see if rotated by specfied amount by examining agent state's quaternion
def test_env_update_angular():
    my_env = habitat_ros.sim_env(env_config_file="configs/tasks/pointnav_rgbd.yaml")
    initial_observations = my_env.observations
    #my_env.start()
    my_env.set_yaw(1)
    my_env.update_orientation()
    post_observations= my_env.observations

    import numpy as np
    assert np.array_equal(initial_observations['depth'],post_observations['depth'])==False

def test_env_update_accel():
    import numpy as np
    my_env = habitat_ros.sim_env(env_config_file="configs/tasks/pointnav_rgbd.yaml")
    my_env.set_dt(1)
    my_env.set_acceleration(2)
    initial_position =  my_env.env._sim._sim.agents[0].state.position
    #my_env.start()
    my_env.set_linear_velocity(1,2)
    my_env.update_orientation()
    post_position = my_env.env._sim._sim.agents[0].state.position
    
    v_actual = my_env.prevv+my_env.acceleration*my_env._dt
    #finish

    d_should_travel = np.sqrt((1*my_env._dt)**2 + (2*my_env._dt)**2)
    assert d_should_travel*0.9<(np.linalg.norm(post_position-initial_position))<d_should_travel*1.1

