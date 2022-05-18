# -*- coding: utf-8 -*-
# @Time    : 2021/12/17 19:28
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : anchors.py
# @Software: PyCharm
import numpy as np


def anchors_python(height, width, stride, base_anchors):
    """
    Parameters
    ----------
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: (A, 4) a base set of anchors
    Returns
    -------
    all_anchors: (height, width, A, 4) ndarray of anchors spreading over the plane
    """
    A = base_anchors.shape[0]
    all_anchors = np.zeros((height, width, A, 4))
    for iw in range(width):
        sw = iw * stride
        for ih in range(height):
            sh = ih * stride
            for k in range(A):
                all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + sw
                all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + sh
                all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + sw
                all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + sh
    return all_anchors


if __name__ == '__main__':
    height = 10
    width = 10
    stride = 2
    base_anchors = np.array([[-8, -8, 23, 23], [0, 0, 15, 15]])
    anchors_python(height, width, stride, base_anchors)
