import numpy as np
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController

class MyFranka:
    def __init__(self, prim_path:str, name:str) -> None:
        self.robot = Franka(prim_path=prim_path, name=name,position=np.array([-0.1, 0,0]))
        self.controller = PickPlaceController(
            name='pick_place_controller',
            gripper=self.robot.gripper,
            robot_articulation=self.robot
        )        
        self.articulation_controller = self.robot.get_articulation_controller()
        self.end_effector_offset = np.array([0, 0.005, 0])
        self.init_flag = False

    def init_step(self):
        if self.init_flag:
            return
        else:
            init_pick = np.array([0.3,0.3,0.5])
            init_place= np.array([0.3,0.3,0.3])
            self.pick_place(init_pick, init_place)
            # while not self.is_done():
            #     pass
            # self.reset()
            # self.init_flag = True
            # print('init finished...')

    def pick_place(self, picking_pos, placing_pos):
        current_joint_stete = self.robot.get_joints_state()
        actions = self.controller.forward(
            picking_position=picking_pos,
            placing_position=placing_pos,
            current_joint_positions=current_joint_stete.positions,
            end_effector_offset=self.end_effector_offset
        )
        self.articulation_controller.apply_action(actions)

    def is_done(self):
        return self.controller.is_done()
    
    def reset(self):
        self.controller.reset()