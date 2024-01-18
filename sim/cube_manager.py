import numpy as np
from omni.isaac.core.objects import DynamicCuboid

ColorMap = {
    "black": [0 ,0, 0],
    "white": [255, 255, 255],
    "red":   [255, 0, 0],
    "green": [0, 255, 0],
    "blue":  [0 ,0, 255]
}

class CubeManager:
    def __init__(self) -> None:
        self.cubes = {}
    def add_cube(self, prim_path, name, position, 
                 scale=[0.05, 0.05, 0.05],
                 color=ColorMap["white"]):
        # create cube
        cube = DynamicCuboid(
            prim_path=prim_path,
            name=name,
            position=np.array(position),
            scale=np.array(scale),
            color=np.array(color)
        )
        # register
        self.cubes[name] = cube
        return cube
    def get_cube(self, name):
        return self.cubes[name]