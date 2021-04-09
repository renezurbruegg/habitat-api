#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from gym import spaces

import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from geometry_msgs.msg import Twist
import threading
import sys

sys.path = [
    b for b in sys.path if "2.7" not in b
]  # remove path's related to ROS from environment or else certain packages like cv2 can't be imported

import habitat
import habitat_sim.bindings as hsim
import magnum as mn
import numpy as np
import time
import cv2

lock = threading.Lock()
rospy.init_node("habitat", anonymous=False)

class sim_env(threading.Thread):

    _x_axis = 0
    _y_axis = 1
    _z_axis = 2
    _dt = 0.00478
    _sensor_rate = 50  # hz
    _r = rospy.Rate(_sensor_rate)

    def __init__(self, cfg):
        threading.Thread.__init__(self)
        self.env = habitat.Env(config=cfg)
        # always assume height equals width
        self._sensor_resolution = {
            "RGB": self.env._sim.config["RGB_SENSOR"]["HEIGHT"],
            "DEPTH": self.env._sim.config["DEPTH_SENSOR"]["HEIGHT"],
            "BC_SENSOR": self.env._sim.config["BC_SENSOR"]["HEIGHT"],
            "SEMANTIC": self.env._sim.config["SEMANTIC_SENSOR"]["HEIGHT"],
        }


        #import pdb; pdb.set_trace();
        self.env._sim._sim.agents[0].move_filter_fn = self.env._sim._sim.step_filter
        self.observations = self.env.reset()

        self.env._sim._sim.agents[0].state.velocity = np.float32([0, 0, 0])
        self.env._sim._sim.agents[0].state.angular_velocity = np.float32([0, 0, 0])

        self._pub_rgb = rospy.Publisher("~rgb", Image, queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", Image, queue_size=1)
        self._pub_semantic_sensor = rospy.Publisher("~semantic", Image, queue_size=1)


        # additional RGB sensor I configured
        self._pub_bc_sensor = rospy.Publisher(
            "~bc_sensor", numpy_msg(Floats), queue_size=1
        )

        self._pub_depth_and_pointgoal = rospy.Publisher(
            "depth_and_pointgoal", numpy_msg(Floats), queue_size=1
        )
        print("created habitat_plant succsefully")

    def _render(self):
        self.env._update_step_stats()  # think this increments episode count
        sim_obs = self.env._sim._sim.get_sensor_observations()
        self.observations = self.env._sim._sensor_suite.get_observations(sim_obs)
        self.observations.update(
            self.env._task.sensor_suite.get_observations(
                observations=self.observations, episode=self.env.current_episode
            )
        )

    def _update_position(self):
        state = self.env.sim.get_agent_state(0)
        vz = -state.velocity[0]
        vx = state.velocity[1]
        dt = self._dt

        start_pos = self.env._sim._sim.agents[0].scene_node.absolute_translation

        ax = (
            self.env._sim._sim.agents[0]
            .scene_node.absolute_transformation()[self._z_axis]
            .xyz
        )
        self.env._sim._sim.agents[0].scene_node.translate_local(ax * vz * dt)

        ax = (
            self.env._sim._sim.agents[0]
            .scene_node.absolute_transformation()[self._x_axis]
            .xyz
        )
        self.env._sim._sim.agents[0].scene_node.translate_local(ax * vx * dt)

        end_pos = self.env._sim._sim.agents[0].scene_node.absolute_translation
        filter_end = self.env._sim._sim.agents[0].move_filter_fn(start_pos, end_pos)
        # Update the position to respect the filter
        self.env._sim._sim.agents[0].scene_node.translate(filter_end - end_pos)
        # self._render()

    def _update_attitude(self):
        """ update agent orientation given angular velocity and delta time"""
        state = self.env.sim.get_agent_state(0)
        yaw = state.angular_velocity[2] / 3.1415926 * 180
        dt = self._dt

        _rotate_local_fns = [
            hsim.SceneNode.rotate_x_local,
            hsim.SceneNode.rotate_y_local,
            hsim.SceneNode.rotate_z_local,
        ]
        _rotate_local_fns[self._y_axis](
            self.env._sim._sim.agents[0].scene_node, mn.Deg(yaw * dt)
        )
        self.env._sim._sim.agents[0].scene_node.rotation = self.env._sim._sim.agents[
            0
        ].scene_node.rotation.normalized()
        # self._render()

    def run(self):
        """Publish sensor readings through ROS on a different thread.
            This method defines what the thread does when the start() method
            of the threading class is called
        """
        while not rospy.is_shutdown():
            lock.acquire()

            # rgb_with_res = np.concatenate(
            #     (
            #         np.float32(self.observations["rgb"].ravel()),
            #         np.array(
            #             [self._sensor_resolution["RGB"], self._sensor_resolution["RGB"]]
            #         ),
            #     )
            # )
            #

            rgb_msg = Image()
            rgb_msg.height = self._sensor_resolution["RGB"]
            rgb_msg.width = self._sensor_resolution["RGB"]
            rgb_msg.data = self.observations["rgb"].flatten().tolist()
            rgb_msg.encoding = "rgb8"
            #
            # # multiply by 10 to get distance in meters
            # depth_with_res = np.concatenate(
            #     (
            #         np.float32(self.observations["depth"].ravel() * 10),
            #         np.array(
            #             [
            #                 self._sensor_resolution["DEPTH"],
            #                 self._sensor_resolution["DEPTH"],
            #             ]
            #         ),
            #     )
            # )

            depth_msg = Image()
            depth_msg.height = self._sensor_resolution["DEPTH"]
            depth_msg.width = self._sensor_resolution["DEPTH"]
            maxDistance = 5 #m
            depth_msg.data = np.uint8((self.observations["depth"].ravel() * 10) * 255/maxDistance).flatten().tolist()
            depth_msg.encoding = "mono8"



            # Create and publish image message
            img_msg = Image()
            img_msg.height = self._sensor_resolution["SEMANTIC"]
            img_msg.width = self._sensor_resolution["SEMANTIC"]
            img_msg.data = self.observations["semantic"].flatten().tolist()
            img_msg.encoding = "mono8"
            # img_pub.publish(img_msg)

            # semantic_with_res = np.concatenate(
            #     (
            #         np.uint32(self.observations["semantic"].ravel()),
            #         np.array(
            #             [
            #                 self._sensor_resolution["SEMANTIC"],
            #                 self._sensor_resolution["SEMANTIC"],
            #             ]
            #         ),
            #     )
            # )


            bc_sensor_with_res = np.concatenate(
                (
                    np.float32(self.observations["bc_sensor"].ravel()),
                    np.array(
                        [
                            self._sensor_resolution["BC_SENSOR"],
                            self._sensor_resolution["BC_SENSOR"],
                        ]
                    ),
                )
            )

            depth_np = np.float32(self.observations["depth"].ravel())
            pointgoal_np = np.float32(self.observations["pointgoal"].ravel())
            lock.release()

            self._pub_rgb.publish(rgb_msg)
            self._pub_depth.publish(depth_msg)
            self._pub_bc_sensor.publish(np.float32(bc_sensor_with_res))

            self._pub_semantic_sensor.publish(img_msg)


            depth_pointgoal_np = np.concatenate((depth_np, pointgoal_np))
            self._pub_depth_and_pointgoal.publish(np.float32(depth_pointgoal_np))
            self._r.sleep()

    def set_linear_velocity(self, vx, vy):
        self.env._sim._sim.agents[0].state.velocity[0] = vx
        self.env._sim._sim.agents[0].state.velocity[1] = vy

    def set_yaw(self, yaw):
        self.env._sim._sim.agents[0].state.angular_velocity[2] = yaw

    def update_orientation(self):
        lock.acquire()
        self._update_attitude()
        self._update_position()
        self._render()
        lock.release()

    def set_dt(self, dt):
        self._dt = dt


def callback(vel, my_env):
    lock.acquire()
    my_env.set_linear_velocity(vel.linear.x, vel.linear.y)
    my_env.set_yaw(vel.angular.z)
    lock.release()




# Define the sensor and register it with habitat
# For the sensor, we will register it with a custom name
@habitat.registry.register_sensor(name="my_supercool_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args, **kwargs):
        return "agent_position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args, **kwargs):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations, episode):
        return self._sim.get_agent_state().position


# Define the sensor and register it with habitat
# For the sensor, we will register it with a custom name
@habitat.registry.register_sensor(name="semantic_segmentation_mask")
class SemanticSegmentationMaskSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args, **kwargs):
        return "semantic_segmentation_mask"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args, **kwargs):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations, episode):
        return self._sim.get_agent_state().position


def main():
    config = habitat.get_config(config_paths="/home/rene/catkin_ws/src/habitat-api/configs/tasks/pointnav_rgbd.yaml")

    config.defrost()

    # Add things to the config to for the measure
    config.TASK.EPISODE_INFO = habitat.Config()
    # The type field is used to look-up the measure in the registry.
    # By default, the things are registered with the class name
    config.TASK.EPISODE_INFO.TYPE = "EpisodeInfo"
    config.TASK.EPISODE_INFO.VALUE = 5

    # Now define the config for the sensor
    config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
    # Use the custom name
    config.TASK.AGENT_POSITION_SENSOR.TYPE = "my_supercool_sensor"
    # Add the sensor to the list of sensors in use
    config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")


    # Now define the config for the sensor
    config.TASK.SEMANTIC_SEGMENTATION_MASK_SENSOR = habitat.Config()
    # Use the custom name
    config.TASK.SEMANTIC_SEGMENTATION_MASK_SENSOR.TYPE = "semantic_segmentation_mask"
    # Add the sensor to the list of sensors in use
    config.TASK.SENSORS.append("SEMANTIC_SEGMENTATION_MASK_SENSOR")

    config.freeze()

    my_env = sim_env(config)
    # start the thread that publishes sensor readings
    my_env.start()

    rospy.Subscriber("/cmd_vel", Twist, callback, (my_env), queue_size=1)
    # define a list capturing how long it took
    # to update agent orientation for past 3 instances
    # TODO modify dt_list to depend on r1
    dt_list = [0.009, 0.009, 0.009]
    while not rospy.is_shutdown():

        start_time = time.time()
        # cv2.imshow("bc_sensor", my_env.observations['bc_sensor'])
        # cv2.waitKey(100)
        # time.sleep(0.1)
        my_env.update_orientation()

        dt_list.insert(0, time.time() - start_time)
        dt_list.pop()
        my_env.set_dt(sum(dt_list) / len(dt_list))


if __name__ == "__main__":
    main()
