from vlm_utils.vlm_client import *

class VLM:
    def __init__(self):

        self.det_model_name = "grounding-dino"
        self.seg_model_name = "segment-anything"
        self.config_path = r"vlm_utils/config.yaml"
        
    def detection(self, image_path, prompt):
        
        image = PI.open(image_path)
        image = np.array(image)
        # print(image.shape)
        det_bbox = det_query(self.det_model_name, image, prompt, self.config_path)
        
        mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        center_x = []
        center_y = []
        # Set values inside each bounding box to 1
        for box in det_bbox:
            x, y, h, w = int(box[0] - box[2] / 2), int(box[1] - box[3] / 2), int(box[2]), int(box[3])
            # print(x,y,h,w)
            # print(mask.shape)
            mask[x:x + h,y:y + w] = 1
            center_x.append(box[0])
            center_y.append(box[1])
        image_draw = draw_bbox(image, obb2poly(np.array(det_bbox)).astype(int))
        # print("det_bbox : ",det_bbox)
        # print(np.sum(mask))
        PI.fromarray(image_draw).save(r'D:\Nvidia-Omniverse\pkg\isaac_sim-2023.1.0-hotfix.1\WorkSpace\AML\sim\result\box.png')
        PI.fromarray(mask.astype(np.uint8) * 255).save(r'D:\Nvidia-Omniverse\pkg\isaac_sim-2023.1.0-hotfix.1\WorkSpace\AML\sim\result\mask.png')
        center_x = [i / image.shape[0] for i in center_x]
        center_y = [i / image.shape[1] for i in center_y]
        # print(center_x,center_y)
        return mask,center_x,center_y
    
    def trans_pos(self, center_x, center_y, pos):
        
        """根据center得到目标坐标函数

        Args:
            center_x (list): 归一化的中心x坐标list 可多目标检测
            center_y (list): 归一化的中心y坐标list 可多目标检测
            pos (list) : [(pos1x,pos1y),(pos2x,pos2y)],输入坐标范围的对角二元组
        """
        pos1x, pos1y = pos[0]
        pos2x, pos2y = pos[1]
        world_H = abs(pos2x - pos1x)
        world_W = abs(pos2y - pos1y)
        
        world_x = []
        for x in center_x:
            world_x.append(min(pos1x, pos2x) + world_H * x)
            
        world_y = []
        for y in center_y:
            world_y.append(min(pos1y, pos2y) + world_W * y)
        
        target_pos = list(zip(world_x, world_y))
        
        return target_pos
    
    def find_object_pos(self, prompt, image_path,pos=None):
        
        """根据物体输入描述,实际的坐标位置,直接返回目标坐标

        Args:
            pos (list) : [(pos1x,pos1y),(pos2x,pos2y)],输入坐标范围的对角二元组
            prompt (str): 输入的物体描述
            image_path (str): 输入图像路径
        """
        if pos is None:
            pos = [(0.7, 0.7), (0.1, 0.1)]
        mask, center_x, center_y = self.detection(image_path=image_path,prompt=prompt)
        target_pos = self.trans_pos(center_x=center_x,center_y=center_y,pos=pos)
        target_pos = target_pos[0]
        x = 0.8-target_pos[0]
        y = 0.8-target_pos[1]
        return [round(x,3),round(y,3)]
        
    
if __name__ == '__main__':
    vlm = VLM()
    # image_path ="D:/Nvidia-Omniverse/pkg/isaac_sim-2023.1.0-hotfix.1/WorkSpace/AML/sim/img/test3.png"
    image_path = "D:/Nvidia-Omniverse/pkg/isaac_sim-2023.1.0-hotfix.1/WorkSpace/AML/background_3.jpg"
    # mask,center_x,center_y = vlm.detection(image_path=image_path,prompt="find the blue flower")
    # print('mask positiobn:',center_x,center_y)
    # 这里世界坐标系用原像素大小测试了,所以应该和上面输出(*image比例因子)一样才对
    target_pos = vlm.find_object_pos(prompt="green bone",image_path=image_path)
    print('target position',target_pos)
    
    # test : python VLM.py
