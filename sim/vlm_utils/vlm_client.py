import grpc
import numpy as np

from .service_pb2 import DetectRequest
from .service_pb2 import Image
from .service_pb2 import Point
from .service_pb2 import SegmentRequest
from .service_pb2_grpc import ImageServiceStub
from .utils import parse_config
from .draw import draw_bbox, obb2poly
from PIL import Image as PI
import argparse


def seg_query(model_name: str, image: np.ndarray, prompt: str, config_path: str = None):
    local_config = parse_config(config_path)

    server = local_config['server']
    x, y = prompt.split()
    x, y = float(x), float(y)
    x, y = int(image.shape[0] * float(x)), int(image.shape[1] * float(y))

    with grpc.insecure_channel(server) as channel:
        stub = ImageServiceStub(channel)
        response = stub.segment(
            SegmentRequest(
                model=model_name,
                prompt=Point(x=y, y=x),  # Segment Anything take reverse coordinate (x-horizontal, y-vertical)
                image=Image(
                    data=image.tobytes(),
                    height=image.shape[0],
                    width=image.shape[1],
                ),
            ))
    mask = np.frombuffer(
        response.mask.data,
        dtype=np.uint8,
    ).reshape(response.mask.height, response.mask.width, -1).astype(bool)

    return mask


def det_query(model_name: str, image: np.ndarray, prompt: str, config_path: str = None):
    local_config = parse_config(config_path)

    server = local_config['server']

    with grpc.insecure_channel(server) as channel:
        stub = ImageServiceStub(channel)
        response = stub.detect(
            DetectRequest(
                model=model_name,
                target=prompt,
                image=Image(
                    data=image.tobytes(),
                    height=image.shape[0],
                    width=image.shape[1],
                ),
            ))
    bboxes = [[i.center.x, i.center.y, i.height_x, i.width_y] for i in response.boxes]
    return bboxes


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, help="Text prompt")
    parser.add_argument('--task', type=str, choices=['det', 'seg'])
    parser.add_argument('--image-path', type=str)
    parser.add_argument('--model-name', type=str, help="Used model name", default='chatglm')
    parser.add_argument('--config-path', type=str, help="Path to Config", default=None)
    args = parser.parse_args()

    image = PI.open(args.image_path)

    if args.task == 'seg':
        seg_mask = seg_query(args.model_name, np.array(image), args.prompt, args.config_path)
        PI.fromarray(seg_mask.squeeze()).save('/Users/zhangkai/Downloads/airbot/tmp/mask.png')
    elif args.task == 'det':
        det_bbox = det_query(args.model_name, np.array(image), args.prompt, args.config_path)
        image_draw = draw_bbox(image, obb2poly(np.array(det_bbox)).astype(int))
        print("det_bbox : ",det_bbox)
        PI.fromarray(image_draw).save('/Users/zhangkai/Downloads/airbot/tmp/box.png')
    else:
        print('unsupported task')
