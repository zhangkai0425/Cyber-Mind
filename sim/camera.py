import numpy as np
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
import os
from PIL import Image

class MyCamera:
    def __init__(self, prim_path, position, freq=20, resolution=(256, 256), orientation=[0, 90, 0]) -> None:
        self.cam = Camera(
            prim_path=prim_path,
            position=np.array(position),
            frequency=freq,
            resolution=resolution,
            orientation=rot_utils.euler_angles_to_quats(np.array(orientation),degrees=True)
        )
        self.cam.initialize()
        self.cam.add_motion_vectors_to_frame()
    def get_img(self):
        img_data = self.cam.get_rgb()
        return img_data
    def save_img(self, folder_path):
        # get image data
        img_data = self.get_img()
        # print(type(img_data))
        # print(img_data.shape)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # np.save(os.path.join(folder_path,"image.npy") , img_data)
        save_path = os.path.join(folder_path, f"frame_now.png")

        pil_img = Image.fromarray(img_data)

        pil_img.save(save_path)

        print(f"Image saved at: {save_path}")
    def get_current_frame(self):
        return self.cam.get_current_frame
    
