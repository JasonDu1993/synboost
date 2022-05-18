# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 16:37
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : draw_box_and_kpt_util.py
# @Software: PyCharm
import numpy  as np
import cv2
import matplotlib.pyplot as plt
from image_dissimilarity.data.cityscapes_labels import trainId2color, trainId2name

obstacle_trainid = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
                    5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain',
                    10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck',
                    15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle',
                    -1: 'unkonwn'}
print("trainId2name", trainId2name)


def draw_total_box(img, rects, thickness=1, fontScale=1, debug=False):
    if isinstance(img, str):
        img = cv2.imread(img)
    else:
        img = np.array(img)
    for i in rects:
        rects_i = rects[i]
        if i == -1:
            color = (0, 0, 255)
            name = "unknown"
        else:
            color = trainId2color[i]
            name = trainId2name[i]

        for rect in rects_i:
            x1, y1, x2, y2 = rect
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)
            cv2.putText(img, name, (max(0, x1), max(0, y1 - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=fontScale, color=color, thickness=thickness)
    if debug:
        plt.imshow(img[:, :, ::-1])
        plt.show()
    return img
