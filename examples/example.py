#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat
import pickle
import matplotlib.pyplot as plt


def example():
    observations_list=[]
    env = habitat.Env(config=habitat.get_config("configs/tasks/pointnav.yaml"))

    print("Environment creation successful")
    observations = env.reset()
    
    print("Agent stepping around inside environment.")
    count_steps = 0
    while not env.episode_over:
        #observations = env.step(env.action_space.sample())

        observations = env.step(0)
        plt.imshow(observations['rgb'])
        plt.show()
        ax=env._sim._sim.agents[0].scene_node.absolute_transformation()[0:3, 2]
        env._sim._sim.agents[0].scene_node.translate_local(ax * 10)

        observations_list.append(observations)
        count_steps += 1
    print("Episode finished after {} steps.".format(count_steps))
    
    #for determining what observation and action spaces are
    print("The action space is: ", env.action_space)
    print("The observation space is: ", env.observation_space)
    pickle_out = open("./examples/from_example_dot_py.pickle", "wb")
    pickle.dump(observations_list,pickle_out)
    pickle_out.close()
    print("pickle was saved")


if __name__ == "__main__":
    example()
