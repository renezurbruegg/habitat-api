#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import os

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
pub_depth_and_pointgoal = rospy.Publisher("depth_and_pointgoal", numpy_msg(Floats), queue_size=10)

rospy.init_node("plant_model", anonymous=True)

class habitat_plant:

    _x_axis = 0
    _y_axis = 1
    _z_axis = 2

    def update_position(self,vz, vx, dt):
        #vx=vx*0.3
        #vz=vz*0.3
        """ update agent position in xz plane given velocity and delta time"""
        start_pos = self.env._sim._sim.agents[0].scene_node.absolute_position()

        ax = self.env._sim._sim.agents[0].scene_node.absolute_transformation()[0:3, self._z_axis]
        self.env._sim._sim.agents[0].scene_node.translate_local(ax * vz * dt)

        ax = self.env._sim._sim.agents[0].scene_node.absolute_transformation()[0:3, self._x_axis]
        self.env._sim._sim.agents[0].scene_node.translate_local(ax * vx * dt)

        end_pos = self.env._sim._sim.agents[0].scene_node.absolute_position()

        # can apply or not apply filter
        filter_end = self.env._sim._sim.agents[0].move_filter_fn(start_pos, end_pos)
        self.env._sim._sim.agents[0].scene_node.translate(filter_end - end_pos)

    def update_attitude(self,roll, pitch, yaw, dt):
        """ update agent orientation given angular velocity and delta time"""
        #roll =0
        #pitch =0
        #yaw=yaw*0.08
        ax_roll = np.zeros(3, dtype=np.float32)
        ax_roll[self._z_axis] = 1
        self.env._sim._sim.agents[0].scene_node.rotate_local(np.deg2rad(roll * dt), ax_roll)
        self.env._sim._sim.agents[0].scene_node.normalize()

        ax_pitch = np.zeros(3, dtype=np.float32)
        ax_pitch[self._x_axis] = 1
        self.env._sim._sim.agents[0].scene_node.rotate_local(
            np.deg2rad(pitch * dt), ax_pitch
        )
        self.env._sim._sim.agents[0].scene_node.normalize()

        ax_yaw = np.zeros(3, dtype=np.float32)
        ax_yaw[self._y_axis] = 1
        self.env._sim._sim.agents[0].scene_node.rotate_local(np.deg2rad(yaw * dt), ax_yaw)
        self.env._sim._sim.agents[0].scene_node.normalize()


    def __init__(self):
        self.env = habitat.Env(
            config=habitat.get_config("configs/tasks/pointnav_rgbd.yaml")
            )
        self.env._sim._sim.agents[0].move_filter_fn = self.env._sim._sim._step_filter
        self.observations = self.env.reset()
        self.vel = np.float32([0,0,0,0])
        print("created object succsefully")


def main():
    bc_plant = habitat_plant()
    flag = 1
   
    while not (bc_plant.env.episode_over or rospy.is_shutdown()):
        print(flag)
        if flag ==1:
            pub_rgb.publish(np.float32(bc_plant.observations["rgb"].ravel()))
            pub_depth.publish(np.float32(bc_plant.observations["depth"].ravel()))#change to not multiply by 10 for eva_baseline to work
            depth_np = np.float32(bc_plant.observations["depth"].ravel())
            pointgoal_np = np.float32(bc_plant.observations['pointgoal'].ravel())
            depth_pointgoal_np = np.concatenate((depth_np,pointgoal_np))
            pub_depth_and_pointgoal.publish(np.float32(depth_pointgoal_np))
            flag = 0
            continue
      
        print('line before wait for message')
        data = rospy.wait_for_message('linear_vel_command', numpy_msg(Floats), timeout=None)
        print('velocity heard is ' + str(bc_plant.vel))
        bc_plant.vel = data.data
        
        print('hab_ros_plant point_goal before update is '+ str(bc_plant.observations['pointgoal'].ravel()))
        
        bc_plant.update_position(bc_plant.vel[0], bc_plant.vel[1], 1)
        bc_plant.update_attitude(0,bc_plant.vel[2],bc_plant.vel[3],1)

        bc_plant.env._update_step_stats() #think this increments episode count
        sim_obs =  bc_plant.env._sim._sim.get_sensor_observations()
        bc_plant.observations = bc_plant.env._sim._sensor_suite.get_observations(sim_obs)
        bc_plant.observations.update(
            bc_plant.env._task.sensor_suite.get_observations(
                observations=bc_plant.observations, episode=bc_plant.env.current_episode
            )
        )

        pub_rgb.publish(np.float32(bc_plant.observations["rgb"].ravel()))
        pub_depth.publish(np.float32(bc_plant.observations["depth"].ravel()))#change to not multiply by 10 for eva_baseline to work
        depth_np = np.float32(bc_plant.observations["depth"].ravel())
        pointgoal_np = np.float32(bc_plant.observations['pointgoal'].ravel())
        depth_pointgoal_np = np.concatenate((depth_np,pointgoal_np))
        pub_depth_and_pointgoal.publish(np.float32(depth_pointgoal_np))   

if __name__ == "__main__":
    main()
