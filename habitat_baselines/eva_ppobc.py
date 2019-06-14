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
from std_msgs.msg import Int32

initial_sys_path = sys.path
sys.path = [b for b in sys.path if "2.7" not in b]
sys.path.insert(0, os.getcwd())

import argparse

import torch
import time

import habitat
from habitat.config.default import get_config
from config.default import get_config as cfg_baseline
import cv2

from train_ppo import make_env_fn
from rl.ppo import PPO, Policy
from rl.ppo.utils import batch_obs
sys.path = initial_sys_path

pub_action = rospy.Publisher("action_id", Int32, queue_size=10)
rospy.init_node('controller_nn', anonymous=True)

def transform_callback(data):
    #print(rospy.get_name(), "Plant heard %s" % str(data.data))
    print("I heard "+str(data.data))

def main():
    rospy.Subscriber("action_id", Int32, transform_callback)
    
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
        default="RGB_SENSOR,DEPTH_SENSOR",
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
    
    foo =     ['--model-path', "/home/bruce/NSERC_2019/habitat-api/data/checkpoints/rgbd.pth", \
    '--sim-gpu-id', '0',\
    '--pth-gpu-id','0', \
    '--num-processes', '1', \
    '--count-test-episodes', '100', \
    '--task-config', "configs/tasks/pointnav_gibson.yaml" ]
    args = parser.parse_args(foo)
    #args = parser.parse_args()

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

#get assign actor critic 
    actor_critic = Policy( 
        observation_space=envs.observation_spaces[0],
        action_space=envs.action_spaces[0],
        hidden_size=512,
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

# convert actor_crtiic to ppo.actor_critic
    actor_critic = ppo.actor_critic

    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    test_recurrent_hidden_states = torch.zeros(
        args.num_processes, args.hidden_size, device=device
    )
    not_done_masks = torch.zeros(args.num_processes, 1, device=device)
    rate = rospy.Rate(10)

#while still evaluating stuff
    while not rospy.is_shutdown():
       
    #get the next action number (actions if more than 1 process) produce actions here
        _, actions, _, test_recurrent_hidden_states = actor_critic.act(
            batch,
            test_recurrent_hidden_states,
            not_done_masks,
            deterministic=False,
        )
        
        pub_action.publish(actions.item())
        
        rate.sleep()


if __name__ == "__main__":
    main()
