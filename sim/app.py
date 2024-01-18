import os

# create app
from omni.isaac.kit import SimulationApp
sim_config = {"headless": False}
simulation_app = SimulationApp(sim_config)

# laod stage
from omni.isaac.core.utils.stage import open_stage
stage_path = "D:/Nvidia-Omniverse/pkg/isaac_sim-2023.1.0-hotfix.1/WorkSpace/AML/task_stage.usd"
open_stage(usd_path=stage_path)
# create world
import numpy as np
from omni.isaac.core import World
my_world = World(stage_units_in_meters=1.0)
# my_world.scene.defaultLight
my_world.scene.add_default_ground_plane()

# create franka
from franka import MyFranka
franka_prim_path = "/World/franka"
franka_name =  "my_franka"
franka = MyFranka(franka_prim_path, franka_name)
my_world.scene.add(
    franka.robot
)
# create camera
from camera import MyCamera
cam_prim_path = "/World/Camera1"
cam_pos = [0.4,0.0,3.5]
cam_freq = 30
cam_reslution = (512, 512)
cam_orientation = [0,90,0]
my_cam = MyCamera(cam_prim_path, cam_pos, cam_freq, cam_reslution, cam_orientation)
cam = my_world.scene.add(
    my_cam.cam
)
# create ChatGLM
from chatglm import ChatGLM,LLM_prompt
my_key = "19fa2ef4dd48412aa45a75dde70d3a21.sJZ7dp76fRRjRDWQ"
glm = ChatGLM(my_key)
# build scene
from cube_manager import CubeManager, ColorMap
cube_manager = CubeManager()
cube1 = my_world.scene.add(
    cube_manager.add_cube("/World/cube1", "cube1", [0.2,-0.2,0.026],color=ColorMap['blue'])
)
cube2 = my_world.scene.add(
    cube_manager.add_cube("/World/cube2", "cube2", [0.25,-0.25,0.026],color=ColorMap['red'])
)
cube3 = my_world.scene.add(
    cube_manager.add_cube("/World/cube3", "cube3", [0.35,-0.35,0.026],color=ColorMap['green'])
)
# preparations   
path = [[0.4, 0.4, 0.36], [0.6,0.6,0.36]]
p = 0
cube_pos, _ = cube1.get_world_pose()
goal_pos = np.array(path[p])

# simulation begin
my_world.reset()
reset = True
while simulation_app.is_running():
    my_world.step(render=True)
    # ----- begin ----- #
    folder_path = r"D:\Nvidia-Omniverse\pkg\isaac_sim-2023.1.0-hotfix.1\WorkSpace\AML\sim\image"
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    if True:
        if my_world.current_time_step_index == 0:
            my_world.reset()
            franka.reset()
        if franka.init_flag:
            franka.pick_place(cube_pos, goal_pos)
        else:
            franka.init_step()
            # reset = True
        if franka.is_done():
            # TODO: input x,y,z

            my_world.pause()
            # p = (p+1)%(len(path))
            # goal_pos = path[p]
            if reset:
                cube_pos = np.array([-0.3,-0.3,0.28])
                goal_pos = np.array([-0.3,-0.3,0.3]) 
                reset = False
            else:
                my_cam.save_img(folder_path = r"D:\Nvidia-Omniverse\pkg\isaac_sim-2023.1.0-hotfix.1\WorkSpace\AML\sim\image")
                # GLM， LVM
                I = input('请输入你的指令:\n')
                # name, [x,y,z] = glm.Chat(I)
                name, [x,y,z] = glm.Decision(I)
                # cube_pos, _ = cube1.get_world_pose()
                cube_object = cube_manager.get_cube(name)
                cube_pos, _ = cube_object.get_world_pose()
                goal_pos = [x,y,z]
                reset = True
            franka.init_flag = True
            my_world.play()
            franka.reset()
            
    # -----  end  ----- #
simulation_app.close()
