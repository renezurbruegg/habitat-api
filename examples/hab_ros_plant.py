#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import os
import threading

PKG = "numpy_tutorial"
import roslib

roslib.load_manifest(PKG)

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

sys.path = [b for b in sys.path if "2.7" not in b]
sys.path.insert(0, os.getcwd())

import habitat
import pickle
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import time

pub_rgb = rospy.Publisher("rgb", numpy_msg(Floats), queue_size=10)
pub_depth = rospy.Publisher("depth", numpy_msg(Floats), queue_size=10)
pub_pose = rospy.Publisher("agent_pose", numpy_msg(Floats), queue_size=10)
pub_depth_and_pointgoal = rospy.Publisher(
    "depth_and_pointgoal", numpy_msg(Floats), queue_size=10
)

rospy.init_node("plant_model", anonymous=True)


class habitat_plant(threading.Thread):

    _x_axis = 0
    _y_axis = 1
    _z_axis = 2

    def __init__(self, env_config_file):
        threading.Thread.__init__(self)
        self.env = habitat.Env(config=habitat.get_config(env_config_file))
        self.env._sim._sim.agents[0].move_filter_fn = self.env._sim._sim._step_filter
        self.observations = self.env.reset()
        self.vel = np.float32([0, 0, 0, 0])
        self.dt = 1
        print("created habitat_plant succsefully")

    def render(self):
        self.env._update_step_stats()  # think this increments episode count
        sim_obs = self.env._sim._sim.get_sensor_observations()
        self.observations = self.env._sim._sensor_suite.get_observations(sim_obs)
        self.observations.update(
            self.env._task.sensor_suite.get_observations(
                observations=self.observations, episode=self.env.current_episode
            )
        )

    def update_position(self):

        vz = self.vel[0]
        vx = self.vel[1]
        dt = self.dt

        start_pos = self.env._sim._sim.agents[0].scene_node.absolute_position()

        ax = self.env._sim._sim.agents[0].scene_node.absolute_transformation()[
            0:3, self._z_axis
        ]
        self.env._sim._sim.agents[0].scene_node.translate_local(ax * vz * dt)

        ax = self.env._sim._sim.agents[0].scene_node.absolute_transformation()[
            0:3, self._x_axis
        ]
        self.env._sim._sim.agents[0].scene_node.translate_local(ax * vx * dt)

        end_pos = self.env._sim._sim.agents[0].scene_node.absolute_position()

        # can apply or not apply filter
        filter_end = self.env._sim._sim.agents[0].move_filter_fn(start_pos, end_pos)
        self.env._sim._sim.agents[0].scene_node.translate(filter_end - end_pos)
        self.render()

    def update_attitude(self):
        """ update agent orientation given angular velocity and delta time"""
        roll = 0  # temporary ban roll and pitch motion
        pitch = 0
        yaw = self.vel[3]
        dt = self.dt

        ax_roll = np.zeros(3, dtype=np.float32)
        ax_roll[self._z_axis] = 1
        self.env._sim._sim.agents[0].scene_node.rotate_local(
            np.deg2rad(roll * dt), ax_roll
        )
        self.env._sim._sim.agents[0].scene_node.normalize()

        ax_pitch = np.zeros(3, dtype=np.float32)
        ax_pitch[self._x_axis] = 1
        self.env._sim._sim.agents[0].scene_node.rotate_local(
            np.deg2rad(pitch * dt), ax_pitch
        )
        self.env._sim._sim.agents[0].scene_node.normalize()

        ax_yaw = np.zeros(3, dtype=np.float32)
        ax_yaw[self._y_axis] = 1
        self.env._sim._sim.agents[0].scene_node.rotate_local(
            np.deg2rad(yaw * dt), ax_yaw
        )
        self.env._sim._sim.agents[0].scene_node.normalize()
        self.render()

    def run(self):
        while not rospy.is_shutdown():
            pub_rgb.publish(np.float32(self.observations["rgb"].ravel()))
            pub_depth.publish(np.float32(self.observations["depth"].ravel()))
            depth_np = np.float32(self.observations["depth"].ravel())
            pointgoal_np = np.float32(self.observations["pointgoal"].ravel())
            depth_pointgoal_np = np.concatenate((depth_np, pointgoal_np))
            pub_depth_and_pointgoal.publish(np.float32(depth_pointgoal_np))
            rospy.sleep(0.005)


def main():
    bc_plant = habitat_plant(env_config_file="configs/tasks/pointnav_rgbd.yaml")
    bc_plant.start() #start a different thread that publishes agent's observations

    while not rospy.is_shutdown():
        
        bc_plant.vel = rospy.wait_for_message("bc_cmd_vel", numpy_msg(Floats), timeout=None).data
        bc_plant.update_position()
        bc_plant.update_attitude()


if __name__ == "__main__":
    main()
