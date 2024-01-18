# import omni libs
from omni.isaac.kit import SimulationApp
# Create a simulation
app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

from omni.isaac.franka import Franka
import omni.isaac.franka.controllers as franka_controller
from omni.isaac.franka.controllers import PickPlaceController
import numpy as np


# Create a world
world = World()
world.scene.add_default_ground_plane()
# Add some objects
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/random_cube",
        name="cube",
        position=np.array([0.3,0.3,0.03]),
        scale=np.array([0.05,0.05,0.05]),
        color=np.array([0,0,0.5])
    )
)
franka = world.scene.add(
    Franka(
        prim_path="/World/fancy_franka",
        name="franka"
    )
)
# set controller
controller = PickPlaceController(
    name = "pick_place_controller",
    gripper=franka.gripper,
    robot_articulation=franka
)
articulation_controller = franka.get_articulation_controller()
path = [[-0.3,-0.3, 0.026], [0.3, -0.3, 0.026], [0.3,0.3,0.026], [0.3, -0.3, 0.026]]
i = 0
p = 0
# define step function
while app.is_running():
    world.step(render=True)
    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()
            controller.reset()
        observations = world.get_observations()
        cube_pos,_ = cube.get_world_pose()
        current_joint_pos = franka.get_joint_positions()
        actions = controller.forward(
            picking_position=cube_pos,
            placing_position=np.array(path[p]),
            current_joint_positions=current_joint_pos,
            end_effector_offset=np.array([0, 0.005, 0]),
        )
        articulation_controller.apply_action(actions)
        if controller.is_done():
            print("done!")
            cube_pos, _ = cube.get_world_pos()
            p = (p+1)%(len(path))
            controller.reset()
        
app.close()
# world.reset()
# try:
#     while True:
#         world.step()
# except KeyboardInterrupt as error:
#     app.close()