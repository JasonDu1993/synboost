# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 15:16
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : pytorch_to_onnx.py
# @Software: PyCharm
import torch
import torch.nn
import os

import numpy as np
from time import time
import torch.onnx
import cv2

from total_utils.seg_to_rect import postprocessing
from total_utils.draw_box_util import draw_total_box


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_jit(save_path, input_shape):
    print("save_jit_path: {}".format(save_path))
    c, h, w = input_shape
    from build_model2 import RoadAnomalyDetector
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = RoadAnomalyDetector(True, input_shape=(c, h, w))

    x = torch.randn(1, c, h, w, requires_grad=False).cuda()
    jit_model = torch.jit.trace(model, x)
    torch.jit.save(jit_model, save_path)


def img_preprocess(image, input_shape):
    # image = Image.open(img_path)
    print("....input")
    c, h, w = input_shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 颜色转换
    img_resize = cv2.resize(img_rgb, (w, h))  # [h, w, c]  , 3 is bgr
    # img = torch.from_numpy(img_np)
    # img = img_resize.transpose((2, 0, 1))[None, ...].astype(np.float32)
    img = img_resize.transpose((2, 0, 1)).astype(np.float32)
    # img = np.stack([img, img])
    img = img[None, ...]
    return img


def run_jit(model_path, input_shape):
    model = torch.jit.load(model_path)
    root = "./sample_images"
    names = list(os.listdir(root))
    names = names
    for i, name in enumerate(names):
        img_path = os.path.join(root, name)
        print("img_path: {}".format(img_path))
        # img_path = "./sample_images/road_anomaly_example.png"
        # img_path = "/zhoudu/workspaces/road_obstacles/sample_images/onnx_conversion/leftImg8bit/val/frankfurt/frankfurt_000001_083852_leftImg8bit.png"
        # mask_path = "/zhoudu/workspaces/road_obstacles/sample_images/onnx_conversion/gtFine/val/frankfurt/frankfurt_000001_083852_gtFine_labelIds.png"
        image = cv2.imread(img_path)
        image_og_h, image_og_w, _ = image.shape
        x = img_preprocess(image, input_shape)
        x = torch.from_numpy(x).cuda()
        print("x shape:", x.shape)
        # for i in range(100):
        # torch.cuda.synchronize()
        t0 = time()
        # print("i:", i)
        outputs = model(x)
        # torch.cuda.synchronize()
        t1 = time()
        print("jit run {} s".format(t1 - t0))
        seg_final, anomaly_score = outputs

        # 可视化diss_pred
        diss_pred_img = anomaly_score[0]

        obs_h, obs_w = diss_pred_img.shape
        # 可视化分割图
        seg_final_img = seg_final[0].astype(np.uint8)
        seg_img = cv2.resize(seg_final_img, (obs_w, obs_h))

        result = postprocessing(np.array(seg_img)[None, :], np.array(diss_pred_img)[None, :], image_og_h, image_og_w)
        img_box = draw_total_box(np.array(image), result[0], debug=True)
        print("spend total {} s".format(time() - t0))


if __name__ == '__main__':
    # save_onnx_path = "./model/synboost_dynamic_batch_hw.onnx"
    save_path = "./model/pytorch_h256w512.pt"
    # save_onnx_path = "./model/optimized_synboost_dynamic_batch_hw_1.onnx"
    # save_onnx_path = "./model/optimized_synboost_dynamic_batch_hw.onnx"
    # save_onnx_path = "./model/synboost_b4_h256w512.onnx"
    # save_onnx_path = "optimized_model.onnx"

    # input_shape = [3, 512, 1024]  # c, h, w
    input_shape = [3, 256, 512]  # c, h, w
    # if not os.path.exists(save_onnx_path):
    get_jit(save_path, input_shape)
    run_jit(save_path, input_shape)
