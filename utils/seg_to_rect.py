# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 15:26
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : seg_to_rect.py
# @Software: PyCharm
import cv2
import numpy as np
from collections import defaultdict
from nms.nms import nms

obstacle_trainid = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
                    5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain',
                    10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck',
                    15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle',
                    -1: 'unkonwn'}


def get_rect_from_segmentation(seg):
    """

    Args:
        seg: 二值化之后的结果, shape: [h,w]
    Returns:
        rects: list[list(x1,y1,x2,y2), ...]
    """
    rects = []
    if cv2.getVersionString() <= "3.4.3.18":
        _, contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        rects.append([x, y, x + w, y + h])
        # cv2.rectangle(seg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    rects = np.array(rects).reshape(-1, 4)
    return rects


def postprocessing(prediction, anomaly_score, clss_num=19):
    """

    Args:
        prediction: shape [B, H=1024, W=2048]
        anomaly_score: shape [B, H=1024, W=2048]
    """
    if prediction.ndim == 4:
        prediction = prediction.squeeze(1)
    if anomaly_score.ndim == 4:
        anomaly_score = anomaly_score.squeeze(1)
    b, h, w = prediction.shape
    results = []

    for i in range(b):
        result = {}
        pred = prediction[i]
        for c in [11, 12]:
            seg = np.where(pred == c, 1, 0).astype(np.uint8)
            rects = get_rect_from_segmentation(seg)
            keeps = nms(rects, thresh=0.5)
            rects = rects[keeps]
            result[c] = rects
        anomaly_sco = anomaly_score[i]
        # anomaly_seg = np.where(anomaly_sco < 0, 1, 0).astype(np.uint8)
        anomaly_seg = np.bitwise_and(np.bitwise_or(pred == 0, pred == 1), anomaly_sco > 127).astype(np.uint8)
        anomaly_rects = get_rect_from_segmentation(anomaly_seg)
        result[-1] = anomaly_rects
        results.append(result)
    return results
