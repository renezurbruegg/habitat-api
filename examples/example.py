#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat
import pickle
import matplotlib.pyplot as plt
import numpy as np


def example():
    #env = habitat.Env(config=habitat.get_config("configs/tasks/pointnav.yaml"))
    env = habitat.Env(config=habitat.get_config("configs/tasks/pointnav_rgbd.yaml"))

    print("Environment creation successful")
    observations = env.reset()
    
    print("Agent stepping around inside environment.")
    count_steps = 0

    _x_axis = 0
    _y_axis = 1
    _z_axis = 2

    def update_position(vz,vx,dt):
        ax=env._sim._sim.agents[0].scene_node.absolute_transformation()[0:3, _z_axis]
        env._sim._sim.agents[0].scene_node.translate_local(ax * vz*dt)

        ax=env._sim._sim.agents[0].scene_node.absolute_transformation()[0:3, _x_axis]
        env._sim._sim.agents[0].scene_node.translate_local(ax * vx*dt)

    def update_attitude(roll,pitch,yaw,dt):

        ax_roll = np.zeros(3, dtype=np.float32)
        ax_roll[_z_axis] = 1
        env._sim._sim.agents[0].scene_node.rotate_local(np.deg2rad(roll*dt), ax_roll)
        env._sim._sim.agents[0].scene_node.normalize()
        
        ax_pitch = np.zeros(3, dtype=np.float32)
        ax_pitch[_x_axis] = 1
        env._sim._sim.agents[0].scene_node.rotate_local(np.deg2rad(pitch*dt), ax_pitch)
        env._sim._sim.agents[0].scene_node.normalize()

        ax_yaw = np.zeros(3, dtype=np.float32)
        ax_yaw[_y_axis] = 1
        env._sim._sim.agents[0].scene_node.rotate_local(np.deg2rad(yaw*dt), ax_yaw)
        env._sim._sim.agents[0].scene_node.normalize()


    
    while not env.episode_over:
        #observations = env.step(env.action_space.sample())
        
        #ax=env._sim._sim.agents[0].scene_node.absolute_transformation()[0:3, _z_axis]
        #env._sim._sim.agents[0].scene_node.translate_local(ax * 0.8)
        #observations = env.step(3)
        
        update_position(-1,0,1)
        #update_attitude(0,0,30,1)
   
        sim_obs=env._sim._sim.get_sensor_observations()
        observations = env._sim._sensor_suite.get_observations(sim_obs)

        plt.imshow(observations['depth'][:,:,0])
        plt.show()

        plt.imshow(observations['rgb'])
        plt.show()
        
        count_steps += 1
        
    
    

    print("Episode finished after {} steps.".format(count_steps))
    
    #for determining what observation and action spaces are


if __name__ == "__main__":
    example()
