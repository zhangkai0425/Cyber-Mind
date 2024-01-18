from typing import Optional, Union

import cv2
import numpy as np


def obb2poly(obb: Union[np.ndarray, list[float]]) -> np.ndarray:
    """Convert oriented bounding boxes to polygons.

    Args:
        obb (list[float] or np.ndarray, ndim=1 or ndim=2): [x_ctr,y_ctr,w,h,angle] in float

    Returns:
        polys (np.ndarray) with size (N, 4, 2)
    """
    if isinstance(obb, list) and isinstance(obb[0], float):
        obb = np.array(obb)
    elif isinstance(obb, list) and isinstance(obb[0], np.ndarray):
        obb = np.vstack(obb)
    if obb.ndim == 1:
        obb = obb[None, :]
    center_x, center_y, height, width = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3]

    p1y, p1x = center_x - height / 2, center_y - width / 2
    p2y, p2x = center_x + height / 2, center_y - width / 2
    p3y, p3x = center_x + height / 2, center_y + width / 2
    p4y, p4x = center_x - height / 2, center_y + width / 2
    return np.vstack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y]).T.reshape(-1, 4, 2)


def draw_bbox(
        image: np.ndarray,
        polys: np.ndarray,
        cates: Optional[list[str]] = None,
        scores: Optional[np.ndarray] = None,
        color: tuple[int, int, int] = (0, 0, 255),
):
    """
    Draw bounding boxes on image.

    *Args*:
        * `image`: `np.ndarray` of shape `(h, w, 3)`. Images are in **BGR format**.
        * `polys`: `np.ndarray` of shape `(n, 4, 2)`, where `n` is the number of boxes.
            Each instance is a polygon in format `[ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]`.
        * `cates`: `list[str] or None` of length `n`, where `n` is the number of boxes.
        * `scores`: `np.ndarray or None` of shape `(n,)`, where `n` is the number of boxes.
        * `color`: `tuple[int, int, int]`, the BGR color of the bounding boxes. Default to red (0, 0, 255)
    *Return*:
        * `image`: `np.ndarray` of shape `(h, w, 3)`. Bounding boxes (and corresponding scores and / or cates) are drawn on the returned image. Images are in **BGR format**.
    """
    if polys.ndim == 2:
        polys = polys[None, ...]
    centers = (polys[:, 0] + polys[:, 2]).astype(int) // 2

    # The np.ndarray input type is not enough.
    image = np.ascontiguousarray(image, dtype=np.uint8)

    for idx, (poly, center) in enumerate(zip(polys, centers)):
        image = cv2.drawContours(image, [poly.astype(int)], -1, color, 1)
        cate = f':{cates[idx]}' if cates is not None else ''
        score = f'@{scores[idx]:.4f}' if scores is not None else ''
        text = f'#{idx}{cate}{score}'

        # image = cv2.putText(image, text, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image
