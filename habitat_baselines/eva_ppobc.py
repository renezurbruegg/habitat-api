#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# LICENSE file in the root directory of this source tree.
import sys
import os 

PKG = "numpy_tutorial"
import roslib

roslib.load_manifest(PKG)

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from geometry_msgs.msg import Twist
import numpy as np





initial_sys_path = sys.path
sys.path = [b for b in sys.path if "2.7" not in b]
sys.path.insert(0, os.getcwd())


import argparse
import torch
import habitat
from habitat.config.default import get_config
from config.default import get_config as cfg_baseline

from train_ppo import make_env_fn
from rl.ppo import PPO, Policy
from rl.ppo.utils import batch_obs
import time




sys.path = initial_sys_path


pub_vel = rospy.Publisher('cmd_vel', Twist,queue_size=1)
rospy.init_node('controller_nn', anonymous=True)

#define some useful global variables
flag = 2
observation = {}
t_prev_update = time.time()

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--sim-gpu-id", type=int, required=True)
    parser.add_argument("--pth-gpu-id", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--count-test-episodes", type=int, default=100)
    parser.add_argument(
        "--sensors",
        type=str,
        default="DEPTH_SENSOR",
        help="comma separated string containing different"
        "sensors to use, currently 'RGB_SENSOR' and"
        "'DEPTH_SENSOR' are supported",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="configs/tasks/pointnav.yaml",
        help="path to config yaml containing information about task",
    )
    
    cmd_line_inputs =     ['--model-path', "/home/bruce/NSERC_2019/habitat-api/data/checkpoints/depth.pth", \
    '--sim-gpu-id', '0',\
    '--pth-gpu-id','0', \
    '--num-processes', '1', \
    '--count-test-episodes', '100', \
    '--task-config', "configs/tasks/pointnav.yaml" ]
    args = parser.parse_args(cmd_line_inputs)


    device = torch.device("cuda:{}".format(args.pth_gpu_id))

    env_configs = []
    baseline_configs = []

    for _ in range(args.num_processes):
        config_env = get_config(config_paths=args.task_config)
        config_env.defrost()
        config_env.DATASET.SPLIT = "val"

        agent_sensors = args.sensors.strip().split(",")
        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

    assert len(baseline_configs) > 0, "empty list of datasets"

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(env_configs, baseline_configs, range(args.num_processes))
            )
        ),
    )

    ckpt = torch.load(args.model_path, map_location=device)

    actor_critic = Policy(
        observation_space=envs.observation_spaces[0],
        action_space=envs.action_spaces[0],
        hidden_size=512,
        goal_sensor_uuid="pointgoal"
    )
    actor_critic.to(device)

    ppo = PPO(
        actor_critic=actor_critic,
        clip_param=0.1,
        ppo_epoch=4,
        num_mini_batch=32,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=2.5e-4,
        eps=1e-5,
        max_grad_norm=0.5,
    )

    ppo.load_state_dict(ckpt["state_dict"])

    actor_critic = ppo.actor_critic

    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    test_recurrent_hidden_states = torch.zeros(
        args.num_processes, args.hidden_size, device=device
    )
    not_done_masks = torch.zeros(args.num_processes, 1, device=device)


    def transform_callback(data):
        nonlocal actor_critic
        nonlocal batch
        nonlocal not_done_masks
        nonlocal test_recurrent_hidden_states
        global flag
        global t_prev_update
        global observation
        
        if flag ==2:
            observation['depth'] =  np.reshape(data.data[0:-2],(256,256,1))
            observation['pointgoal'] = data.data[-2:]
            flag =1
            return 

        pointgoal_received = data.data[-2:]
        translate_amount = 0.25 #meters
        rotate_amount = 0.174533 #radians

        isrotated =  rotate_amount*0.95<=abs(pointgoal_received[1]-observation['pointgoal'][1])<=rotate_amount*1.05
        istimeup = (time.time()-t_prev_update)>=4

        #print('istranslated is '+ str(istranslated))
        #print('isrotated is '+ str(isrotated))
        #print('istimeup is '+ str(istimeup))

        if (isrotated or istimeup):
            vel_msg = Twist()
            vel_msg.linear.x = 0
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = 0
            pub_vel.publish(vel_msg)
            time.sleep(0.2)
            print('entered update step')

            # cv2.imshow("Depth", observation['depth'])
            # cv2.waitKey(100)
            
            observation['depth'] =  np.reshape(data.data[0:-2],(256,256,1))
            observation['pointgoal'] = data.data[-2:]
            
            batch = batch_obs([observation])
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)
            if flag ==1:
                not_done_masks = torch.tensor(
                    [0.0] ,
                    dtype=torch.float,
                    device=device,
                )
                flag = 0
            else:
                not_done_masks = torch.tensor(
                    [1.0] ,
                    dtype=torch.float,
                    device=device,
                )

            _, actions, _, test_recurrent_hidden_states= actor_critic.act(
                batch,
                test_recurrent_hidden_states,
                not_done_masks,
                deterministic=True,
            )
            
            action_id = actions.item()
            print('observation received to produce action_id is '+str(observation['pointgoal']))
            print("action_id from net is "+str(actions.item()))
    
            t_prev_update = time.time()
            vel_msg = Twist()
            vel_msg.linear.x = 0
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = 0
            if action_id == 0:
                vel_msg.linear.x = 0.25/4
                pub_vel.publish(vel_msg)
            elif action_id == 1:
                vel_msg.angular.z = 10/180*3.1415926
                pub_vel.publish(vel_msg)
            elif action_id ==2:
                vel_msg.angular.z = -10/180*3.1415926
                pub_vel.publish(vel_msg)
            else:
                pub_vel.publish(vel_msg)
                sub.unregister()
                print('NN finished navigation task')
            
        
    sub = rospy.Subscriber("depth_and_pointgoal", numpy_msg(Floats), transform_callback,queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
