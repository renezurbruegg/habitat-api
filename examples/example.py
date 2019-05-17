#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat
import pickle
import matplotlib.pyplot as plt


def example():
    #env = habitat.Env(config=habitat.get_config("configs/tasks/pointnav.yaml"))
    env = habitat.Env(config=habitat.get_config("configs/tasks/pointnav_rgbd.yaml"))

    print("Environment creation successful")
    observations = env.reset()
    
    print("Agent stepping around inside environment.")
    count_steps = 0
    while not env.episode_over:
        #observations = env.step(env.action_space.sample())



        #ax=env._sim._sim.agents[0].scene_node.absolute_transformation()[0:3, 2]
        #env._sim._sim.agents[0].scene_node.translate_local(ax * 0.05)
        ax=env._sim._sim.agents[0].scene_node.absolute_transformation()[0:3, 0]
        env._sim._sim.agents[0].scene_node.translate_local(ax * 0.8)
        
        

        #observations = env.step(3)
        sim_obs=env._sim._sim.get_sensor_observations()
        observations = env._sim._sensor_suite.get_observations(sim_obs)
        plt.imshow(observations['rgb'])
        plt.show()

        count_steps += 1
    print("Episode finished after {} steps.".format(count_steps))
    
    #for determining what observation and action spaces are


if __name__ == "__main__":
    example()
