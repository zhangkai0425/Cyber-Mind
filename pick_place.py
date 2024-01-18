# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.tasks import PickPlace

my_world = World(stage_units_in_meters=1.0)
my_task = PickPlace()
my_world.add_task(my_task)
my_world.reset()
task_params = my_task.get_params()
my_franka = my_world.scene.get_object(task_params["robot_name"]["value"])
my_controller = PickPlaceController(
    name="pick_place_controller", gripper=my_franka.gripper, robot_articulation=my_franka
)
articulation_controller = my_franka.get_articulation_controller()
path = [[-0.3,-0.3, 0.026], [0.3, -0.3, 0.026], [0.3,0.3,0.026], [0.3, -0.3, 0.026]]
i = 0
p = 0
cube = my_world.scene.get_object(task_params["cube_name"]["value"])
cube_pos, _  = cube.get_world_pose()
goal_pos = np.array(path[p])
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
        observations = my_world.get_observaons()
        actions = my_controller.forward(
            # picking_position=observations[task_params["cube_name"]["value"]]["position"],
            picking_position=cube_pos,
            placing_position=goal_pos,
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0.005, 0]),
        )
        articulation_controller.apply_action(actions)
        if my_controller.is_done():
            print("done picking and placing")
            # print(observations[task_params["cube_name"]["value"]]["target_position"])
            cube_pos, _ = cube.get_world_pose()
            p = (p+1)%len(path)
            goal_pos = np.array(path[p])
            my_controller.reset()
simulation_app.close()
