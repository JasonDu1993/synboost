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


def filter_rect(rects: np.ndarray):
    delete_num = []
    rect_num = rects.shape[0]
    for i in range(rect_num - 1):
        for j in range(i + 1, rect_num):
            rect_i = rects[i]
            xi1, yi1, xi2, yi2 = rect_i
            rect_j = rects[j]
            xj1, yj1, xj2, yj2 = rect_j

            area_i = (xi2 - xi1) * (yi2 - yi1)
            area_j = (xj2 - xj1) * (yj2 - yj1)

            min_area = min(area_i, area_j)
            l = max(xi1, xj1)
            r = min(xi2, xj2)
            t = max(yi1, yj1)
            b = min(yi2, yj2)
            if l >= r or t >= b:
                continue
            inter_area = (r - l) * (b - t)
            ratio = inter_area / (min_area + 1e-7)
            if ratio > 0.4:
                if area_i > area_j:
                    delete_num.append(j)
                else:
                    delete_num.append(i)
    return delete_num


def get_rect_from_segmentation(seg, origin_h, origin_w):
    """

    Args:
        seg: 二值化之后的结果, shape: [h,w]
    Returns:
        rects: ndarray[[x1,y1,x2,y2], ...], shape [N, 4], N表示框的个数
    """
    seg_h, seg_w = seg.shape
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

        x2 = x + w
        y2 = y + h
        x = int(x * origin_w / (seg_w + 1e-7))
        y = int(y * origin_h / (seg_h + 1e-7))
        x2 = int(x2 * origin_w / (seg_w + 1e-7))
        y2 = int(y2 * origin_h / (seg_h + 1e-7))

        rects.append([x, y, x2, y2])
        # cv2.rectangle(seg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    rects = np.array(rects).reshape(-1, 4)
    rect_num = rects.shape[0]
    delete_num = filter_rect(rects)

    retain_index = []
    for i in range(rect_num):
        if i not in delete_num:
            retain_index.append(i)
    rects = rects[retain_index]
    return rects


def postprocessing(prediction, anomaly_score, origin_h, origin_w):
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
    nums = [0]
    cats = [11, 12, -1]
    for i in range(b):
        result = {}
        pred = prediction[i]
        total_rect = []
        for c in [11, 12]:
            seg = np.where(pred == c, 1, 0).astype(np.uint8)
            rects = get_rect_from_segmentation(seg, origin_h, origin_w)
            result[c] = rects
            total_rect.append(rects)
            nums.append(nums[-1] + rects.shape[0])
        anomaly_sco = anomaly_score[i]
        # anomaly_seg = np.where(anomaly_sco < 0, 1, 0).astype(np.uint8)
        anomaly_seg = np.bitwise_and(np.bitwise_or(pred == 0, pred == 1), anomaly_sco > 127).astype(np.uint8)
        anomaly_rects = get_rect_from_segmentation(anomaly_seg, origin_h, origin_w)
        result[-1] = anomaly_rects
        total_rect.append(anomaly_rects)
        nums.append(nums[-1] + anomaly_rects.shape[0])

        total_rect = np.concatenate(total_rect, axis=0)
        delete_num = filter_rect(total_rect)
        for i in range(len(nums[:-1])):
            retain_num = []
            for j in range(nums[i], nums[i + 1]):
                if j not in delete_num:
                    retain_num.append(j - nums[i])
            c = cats[i]
            retain_rect = result[c][retain_num]
            print_rect = retain_rect.copy()
            print_rect[:, 2] = print_rect[:, 2] - print_rect[:, 0]
            print_rect[:, 3] = print_rect[:, 3] - print_rect[:, 1]
            print("cat:", c, "box", print_rect)
            result[c] = retain_rect

        results.append(result)
    return results
